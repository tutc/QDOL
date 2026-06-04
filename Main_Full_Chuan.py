"""
run_single_experiment.py
========================
Lấy từ ablation_herding_vs_kdpp.py (v4), giữ nguyên toàn bộ logic gốc.
Chỉ thay đổi:
  - Xóa: RandomSelector, compare_exemplar_diversity, run_ablation,
          _print_delta_table, _interpret, toàn bộ Excel export
  - Thêm: hàm run_single() gọi đúng run_one() như file gốc

Các method được hỗ trợ:
  'herding'  → iCaRL Herding
  'kdpp'     → k-DPP α=0
  'qd_off'   → k-DPP QD Offline
  'qdol'     → k-DPP QD α=1 (mặc định)

Sử dụng:
  run_single()
  run_single(dataset='core50', step=2, method='qdol', budget=1000)
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random
from datetime import datetime

import Benchmarks as benchmarks
from kdpp_selector    import KDPPSelector
from herding_selector import HerdingSelector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed   = 317


# =============================================================
# Reproducibility
# =============================================================

def set_seed():
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# =============================================================
# KDPPSelectorOffline
# =============================================================

class KDPPSelectorOffline(KDPPSelector):
    """
    Offline variant: full greedy QR recompute trên M+1 mẫu mỗi bước.
    """

    def __init__(self, M: int, metric: str = "cosine",
                 device: str = "cuda", alpha: float = 1.0):
        super().__init__(M=M, metric=metric, device=device, alpha=alpha)

    def _incremental_update(self, new_feat: torch.Tensor) -> bool:
        candidates        = torch.cat([self.exemplars, new_feat.unsqueeze(0)], dim=0)
        selected_idxs, Q_new, scores_new = self._greedy_qr(candidates)
        self.exemplars    = candidates[selected_idxs].clone()
        self.Q            = Q_new
        self.scores       = scores_new
        return True


# =============================================================
# Generic Main — nhận SelectorClass làm tham số
# =============================================================

class MainGeneric(nn.Module):

    def __init__(self, n_mini_batch, SelectorClass, selector_kwargs=None,
                 n_class=10, n_features=512, metric="cosine"):
        super().__init__()
        self.n_features      = n_features
        self.n_class         = n_class
        self.metric          = metric
        self.n_mini_batch    = n_mini_batch
        self.SelectorClass   = SelectorClass
        self.selector_kwargs = selector_kwargs or {}
        self.memorySize      = 5000
        self.avg_acc         = []
        self.avg_acc_activ   = False
        self.reset()

    def reset(self):
        self.ProtoMatrix       = torch.zeros(self.n_class, self.n_features, device=device)
        self.class_to_exemplar: dict[int, torch.Tensor] = {}
        self.class_to_selector: dict[int, object]       = {}
        self.classes_seen:      list[int]               = []

    def _make_selector(self, M):
        return self.SelectorClass(M=M, metric=self.metric,
                                  device=str(device), **self.selector_kwargs)

    def _update_proto(self, class_id):
        feats = self.class_to_exemplar.get(class_id)
        if feats is None or feats.shape[0] == 0:
            return
        if self.metric == "cosine":
            self.ProtoMatrix[class_id] = F.normalize(feats.mean(dim=0), dim=0)
        else:
            self.ProtoMatrix[class_id] = feats.mean(dim=0)

    def _current_M(self):
        return max(1, self.memorySize // max(1, len(self.classes_seen)))

    def _trim_all_classes(self):
        M_new = self._current_M()
        for class_id in self.classes_seen[:-1]:
            feats = self.class_to_exemplar.get(class_id)
            if feats is None or feats.shape[0] <= M_new:
                continue
            sel = self._make_selector(M_new)
            trimmed, _ = sel.K_DPP(feats)
            self.class_to_exemplar[class_id] = trimmed
            self.class_to_selector[class_id] = sel
            self._update_proto(class_id)

    def storeFeature(self, point, class_id):
        feat = point.to(device).float()
        if self.metric == "cosine":
            feat = F.normalize(feat, dim=0)
        feat_2d = feat.unsqueeze(0)

        if class_id not in self.class_to_exemplar:
            self.classes_seen.append(class_id)
            M_new = self._current_M()
            self._trim_all_classes()
            self.class_to_exemplar[class_id] = feat_2d
            self.class_to_selector[class_id] = self._make_selector(M_new)
        else:
            M_cur = self._current_M()
            sel   = self.class_to_selector[class_id]
            if sel.M != M_cur:
                sel_new = self._make_selector(M_cur)
                trimmed, _ = sel_new.K_DPP(self.class_to_exemplar[class_id])
                self.class_to_exemplar[class_id] = trimmed
                self.class_to_selector[class_id] = sel_new
                sel = sel_new
            self.class_to_exemplar[class_id] = torch.cat(
                [self.class_to_exemplar[class_id], feat_2d], dim=0)
            exemplars, _ = sel.K_DPP(self.class_to_exemplar[class_id])
            self.class_to_exemplar[class_id] = exemplars

        self._update_proto(class_id)

    def updateMemoryBank(self):
        for class_id in self.classes_seen:
            self._update_proto(class_id)

    def forward(self, inputs):
        with torch.no_grad():
            x = inputs.to(device).float()
            if self.metric == "cosine":
                x = F.normalize(x, dim=1)
                return torch.matmul(x, self.ProtoMatrix.T)
            else:
                dot  = torch.matmul(x, self.ProtoMatrix.T)
                p_sq = (self.ProtoMatrix ** 2).sum(dim=1)
                return 2.0 * dot - p_sq

    # =========================================================
    # RNG snapshot helpers
    # =========================================================

    @staticmethod
    def _save_rng():
        return {
            "torch":  torch.get_rng_state(),
            "cuda":   torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "numpy":  np.random.get_state(),
            "python": random.getstate(),
        }

    @staticmethod
    def _restore_rng(state):
        torch.set_rng_state(state["torch"])
        if state["cuda"] is not None:
            torch.cuda.set_rng_state(state["cuda"])
        np.random.set_state(state["numpy"])
        random.setstate(state["python"])

    # =========================================================
    def _eval_task(self, test_features, task_id: int) -> float:
        """Eval thuần 1 task — không đo thời gian, không thay đổi RNG."""
        correct = total = 0
        with torch.no_grad():
            for inputs, targets in test_features[task_id]:
                inputs  = inputs.to(device)
                targets = targets.long().to(device)
                preds   = torch.argmax(self.forward(inputs), dim=1)
                correct += (preds == targets).sum().item()
                total   += targets.size(0)
        return correct / total * 100 if total > 0 else 0.0

    def test_idx(self, test_features, idx_test):
        """
        Tính overall accuracy (weighted) trên tập hợp nhiều task trong idx_test.
        Trả về (overall_acc, last_task_acc) — giống hệt file main_v3b.
        Dùng để tính avg_acc (overall acc qua tất cả tasks đã thấy).
        """
        correct = total = 0
        curr_correct = curr_total = 0
        with torch.no_grad():
            for idx in idx_test:
                for inputs, targets in test_features[idx]:
                    inputs  = inputs.to(device)
                    targets = targets.long().to(device)
                    preds   = torch.argmax(self.forward(inputs), dim=1)
                    corr    = (preds == targets).sum().item()
                    curr_correct += corr
                    curr_total   += targets.size(0)
                    correct      += corr
                    total        += targets.size(0)
        overall_acc = correct / total * 100 if total > 0 else 0.0
        last_acc    = curr_correct / curr_total * 100 if curr_total > 0 else 0.0
        return overall_acc, last_acc

    # =========================================================
    def train_test(self, train_features, test_features):
        n_tasks     = len(train_features)
        acc_history = torch.full((n_tasks, n_tasks), float("nan"))
        acc_all     = torch.zeros(n_tasks)
        idx_seen    = []
        self.avg_acc = []
        train_elapsed = 0.0

        with torch.no_grad():
            for task_id, train_loader in enumerate(train_features):
                idx_seen.append(task_id)

                # ── Bước 1: Prefetch (ngoài train timer) ─────────────────
                batches = []
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    if batch_idx == self.n_mini_batch:
                        break
                    batches.append((
                        inputs.to(device),
                        targets.long().to(device)
                    ))

                # ── Bước 2: Train ─────────────────────────────────────────
                t_train = time.perf_counter()
                for inputs, targets in batches:
                    for i in range(inputs.size(0)):
                        self.storeFeature(inputs[i], int(targets[i]))
                self.updateMemoryBank()
                train_elapsed += time.perf_counter() - t_train

                # ── Bước 3: Eval giữa chừng (RNG-isolated) ───────────────
                rng_state = self._save_rng()
                for seen_id in idx_seen:
                    acc_history[task_id, seen_id] = self._eval_task(
                        test_features, seen_id
                    )
                if self.avg_acc_activ:
                    overall_acc, _ = self.test_idx(test_features, idx_seen)
                    self.avg_acc.append(overall_acc)
                self._restore_rng(rng_state)

            # ── Bước 4: Final eval — đo infer time ───────────────────────
            all_test_batches = []
            for task_id in range(n_tasks):
                task_batches = []
                for inputs, targets in test_features[task_id]:
                    task_batches.append((
                        inputs.to(device),
                        targets.long().to(device)
                    ))
                all_test_batches.append(task_batches)

            t_infer = time.perf_counter()
            for task_id in range(n_tasks):
                correct = total = 0
                for inputs, targets in all_test_batches[task_id]:
                    preds    = torch.argmax(self.forward(inputs), dim=1)
                    correct += (preds == targets).sum().item()
                    total   += targets.size(0)
                acc_all[task_id] = correct / total * 100
            infer_elapsed = time.perf_counter() - t_infer

        return acc_history, acc_all, train_elapsed, infer_elapsed

    # =========================================================
    def run_experiment(self, n_mini_batch, train_features, test_features,
                       N_try=5, random_ordering=True):
        avg_per_trial        = torch.zeros(N_try)
        last_per_trial       = torch.zeros(N_try)
        forgetting_per_trial = torch.zeros(N_try)
        train_time_trials    = []
        infer_time_trials    = []
        self.n_mini_batch    = n_mini_batch

        for idx_try in tqdm(range(N_try), desc="  Trials", leave=False):
            self.reset()
            acc_history, acc_all, train_t, infer_t = self.train_test(
                train_features, test_features
            )

            train_time_trials.append(train_t)
            infer_time_trials.append(infer_t)

            # ── Avg Accuracy ─────────────────────────────────────────────
            if self.avg_acc_activ:
                avg_per_trial[idx_try] = float(np.mean(self.avg_acc))

            # ── Last Accuracy ─────────────────────────────────────────────
            last_per_trial[idx_try] = acc_all.mean().item()

            # ── Forgetting (công thức max, bỏ task cuối) ─────────────────
            n_tasks = acc_history.shape[0]
            if n_tasks > 1:
                forgetting_list = []
                for i in range(n_tasks - 1):       # bỏ task cuối
                    hist_i = acc_history[i:, i]
                    valid  = hist_i[~torch.isnan(hist_i)]
                    if valid.numel() == 0:
                        continue
                    peak_i  = valid.max().item()
                    final_i = acc_all[i].item()
                    forgetting_list.append(peak_i - final_i)
                forgetting_per_trial[idx_try] = (
                    float(np.mean(forgetting_list)) if forgetting_list else float("nan")
                )
            else:
                forgetting_per_trial[idx_try] = float("nan")

            if random_ordering:
                pairs = list(zip(train_features, test_features))
                random.shuffle(pairs)
                train_features, test_features = zip(*pairs)

        return (avg_per_trial, last_per_trial, forgetting_per_trial,
                train_time_trials, infer_time_trials)


# =============================================================
# Runner — GIỮ NGUYÊN HÀM run_one() TỪ FILE GỐC
# =============================================================

def run_one(data, SelectorClass, selector_kwargs, memorysize,
            metric, n_mini_batch=55, N_try=5):
    """Chạy 1 cấu hình, trả về mean ± std qua N_try trials."""
    set_seed()
    exp = MainGeneric(
        n_mini_batch    = n_mini_batch,
        SelectorClass   = SelectorClass,
        selector_kwargs = selector_kwargs,
        n_class         = data.n_classes,
        n_features      = data.n_features,
        metric          = metric,
    )
    exp.memorySize    = memorysize
    exp.avg_acc_activ = True

    start = time.time()
    avg_t, last_t, fgt_t, train_times, infer_times = exp.run_experiment(
        n_mini_batch, data.train_features, data.test_features,
        N_try=N_try, random_ordering=True,
    )
    elapsed = round(time.time() - start, 2)

    def ms(t):
        arr = np.array(t) if isinstance(t, list) else t.numpy()
        return round(float(arr.mean()), 2), round(float(arr.std()), 2)

    avg_m,  avg_s  = ms(avg_t)
    last_m, last_s = ms(last_t)
    fgt_m,  fgt_s  = ms(fgt_t)
    tr_m,   tr_s   = ms(train_times)
    inf_m,  inf_s  = ms(infer_times)
    mem = int(sum(f.shape[0] for f in exp.class_to_exemplar.values()))

    return (avg_m, avg_s, last_m, last_s, fgt_m, fgt_s,
            tr_m, tr_s, inf_m, inf_s, mem, elapsed)


# =============================================================
# run_single — hàm mới, gọi đúng run_one() như file gốc
# =============================================================

def run_single(
    dataset:      str = "core50",
    step:         int = 2,
    method:       str = "qdol",
    budget:       int = 1000,
    metric:       str = "cosine",
    n_mini_batch: int = 55,
    N_try:        int = 5,
):
    """
    Chạy một cấu hình với N_try trials và in kết quả ra màn hình.

    Tham số
    -------
    dataset      : 'core50' | 'cub200' | 'cifar100'          (mặc định: 'core50')
    step         : số class mỗi task (cho cub200 / cifar100)  (mặc định: 2)
    method       : 'herding' | 'kdpp' | 'qd_off' | 'qdol'    (mặc định: 'qdol')
    budget       : tổng memory budget                          (mặc định: 1000)
    metric       : 'cosine' | 'euclidean'                     (mặc định: 'cosine')
    n_mini_batch : số mini-batch lấy mỗi task                 (mặc định: 55)
    N_try        : số trials để tính mean ± std               (mặc định: 5)
    """

    # ── Ánh xạ method → SelectorClass + kwargs ────────────────
    METHOD_MAP = {
        "herding": (HerdingSelector,     {},              "Herding"),
        "kdpp":    (KDPPSelector,         {"alpha": 0.0}, "k-DPP α=0"),
        "qd_off":  (KDPPSelectorOffline,  {"alpha": 1.0}, "k-DPP QD Offline"),
        "qdol":    (KDPPSelector,         {"alpha": 1.0}, "k-DPP QD α=1 (Online)"),
    }

    key = method.lower().strip()
    if key not in METHOD_MAP:
        raise ValueError(
            f"Method '{method}' không hợp lệ. "
            f"Chọn một trong: {list(METHOD_MAP.keys())}."
        )
    SelectorClass, sel_kwargs, method_name = METHOD_MAP[key]

    # ── Load dataset ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Đang tải dataset: {dataset.upper()}  (step={step})")
    ds = dataset.lower().strip()
    if ds == "core50":
        origin_dataset = benchmarks.Core50Resnet18_Claude.CORE50RESNET18()
    elif ds == "cub200":
        origin_dataset = benchmarks.Cub200Resnet50.CUB200RESNET50(start=step, step=step)
    elif ds == "cifar100":
        origin_dataset = benchmarks.Cifar100Resnet50.CIFAR100RESNET50(start=step, step=step)
    else:
        raise ValueError(
            f"Dataset '{dataset}' không được hỗ trợ. "
            f"Chọn một trong: 'core50', 'cub200', 'cifar100'."
        )
    print(f"  Classes: {origin_dataset.n_classes}  |  Features: {origin_dataset.n_features}")

    # ── In cấu hình ────────────────────────────────────────────
    print(f"{'─'*60}")
    print(f"  Method  : {method_name}")
    print(f"  Budget  : {budget:,}")
    print(f"  Metric  : {metric}")
    print(f"  N_try   : {N_try}  trials")
    print(f"  Device  : {device}")
    print(f"{'='*60}")

    # ── Chạy — gọi đúng run_one() như file gốc ────────────────
    # data.clone() giống hệt cách run_ablation() gọi run_one() trong file gốc
    data = origin_dataset.clone()
    (avg_m, avg_s, last_m, last_s, fgt_m, fgt_s,
     tr_m, tr_s, inf_m, inf_s, mem_cnt, elapsed) = run_one(
        data, SelectorClass, sel_kwargs,
        memorysize   = budget,
        metric       = metric,
        n_mini_batch = n_mini_batch,
        N_try        = N_try,
    )

    # ── In kết quả ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  KẾT QUẢ  ({N_try} trials)")
    print(f"{'─'*60}")
    print(f"  {'Average Accuracy':<20}: {avg_m:>6.2f} ± {avg_s:.2f}  %")
    print(f"  {'Last Accuracy':<20}: {last_m:>6.2f} ± {last_s:.2f}  %")
    print(f"  {'Forgetting':<20}: {fgt_m:>6.2f} ± {fgt_s:.2f}  %")
    print(f"{'─'*60}")
    print(f"  {'Train Time':<20}: {tr_m:>6.2f} ± {tr_s:.2f}  s")
    print(f"  {'Infer Time':<20}: {inf_m:>6.3f} ± {inf_s:.3f}  s")
    print(f"  {'Exemplars stored':<20}: {mem_cnt:,}")
    print(f"  {'Wall-clock':<20}: {elapsed}  s")
    print(f"{'='*60}\n")

    return {
        "avg_acc":    (avg_m,  avg_s),
        "last_acc":   (last_m, last_s),
        "forgetting": (fgt_m,  fgt_s),
        "train_time": (tr_m,   tr_s),
        "infer_time": (inf_m,  inf_s),
    }


# =============================================================
# Entry point
# =============================================================

if __name__ == "__main__":
    run_single(
        dataset = "core50",   # 'core50' | 'cub200' | 'cifar100'
        step    = 2,
        method  = "qdol",     # 'herding' | 'kdpp' | 'qd_off' | 'qdol'
        budget  = 1000,
    )
