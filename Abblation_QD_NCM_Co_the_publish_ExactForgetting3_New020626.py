"""
ablation_herding_vs_kdpp.py  (v4 — fixed forgetting + fixed avg_acc)
==========================================
Ablation study: iCaRL Herding vs k-DPP variants vs Random

Các phương pháp so sánh:
  1. Herding          — iCaRL herding (prototype-closest)
  2. Random           — reservoir sampling, baseline tối giản
  3. k-DPP α=0        — greedy k-DPP thuần, chỉ diversity
  4. k-DPP QD α=1     — phương pháp đề xuất (online incremental)
  5. k-DPP QD Offline — như α=1 nhưng recompute scores toàn bộ M+1 mẫu

Fixes so với v4 gốc:
  [FIX 4] Forgetting dùng công thức max (Lopez-Paz 2017):
            Forgetting_i = max_{step=i..T-1}(acc_history[step, i]) − acc_all[i]
          Bỏ task cuối (luôn = 0) trước khi tính mean.
          acc_history (T,T) lưu acc từng task tại mỗi step học.
  [FIX 5] Tất cả eval (acc_history, acc_all) nằm NGOÀI vùng đo
          train_elapsed và infer_elapsed.
  [FIX 6] infer_elapsed chỉ đo final eval (sau khi học hết tất cả tasks),
          không tính eval giữa chừng.
  [FIX 7] train_elapsed không tính .to(device) của data loader (I/O overhead).
  [FIX 8] avg_acc tính bằng overall weighted accuracy qua tất cả tasks đã thấy
          (test_idx gọi trên idx_seen) — KHÔNG dùng nanmean(acc_history[step, :])
          vì nanmean là unweighted mean và cho kết quả khác khi các tasks có số
          mẫu khác nhau. Đồng nhất hoàn toàn với main_v3b_ncm_only.
          Thêm method test_idx() vào MainGeneric để tính weighted overall acc.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random
from datetime import datetime

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

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
# RandomSelector
# =============================================================

class RandomSelector:
    """
    Baseline tối giản: reservoir sampling (Vitter, 1985).
    API giống KDPPSelector để dùng chung trong MainGeneric.
    """

    def __init__(self, M: int, metric: str = "cosine",
                 device: str = "cuda", **kwargs):
        if M <= 0:
            raise ValueError(f"M phải > 0, nhận được M={M}")
        self.M       = M
        self.metric  = metric
        self.device  = device
        self._n_seen = 0
        self.exemplars = None

    def reset(self):
        self._n_seen   = 0
        self.exemplars = None

    def _prep_feats(self, feats: torch.Tensor) -> torch.Tensor:
        feats = feats.to(self.device).float()
        if self.metric == "cosine":
            feats = F.normalize(feats, dim=1)
        return feats

    def K_DPP(self, all_feats: torch.Tensor):
        all_feats = self._prep_feats(all_feats)
        N, d      = all_feats.shape

        if N <= self.M:
            self._n_seen   = N
            self.exemplars = all_feats.clone()
            return self.exemplars.clone(), torch.empty((0, d), device=self.device)

        if self.exemplars is None:
            perm           = torch.randperm(N, device=self.device)[:self.M]
            self.exemplars = all_feats[perm].clone()
            self._n_seen   = N
            mask           = torch.ones(N, dtype=torch.bool, device=self.device)
            mask[perm]     = False
            return self.exemplars.clone(), all_feats[mask]

        if N != self.M + 1:
            raise ValueError(
                f"Incremental mode yêu cầu N = M+1 = {self.M + 1}, nhận N={N}."
            )
        self._n_seen += 1
        j = random.randint(0, self._n_seen - 1)
        if j < self.M:
            self.exemplars       = self.exemplars.clone()
            self.exemplars[j]    = all_feats[-1]
        return self.exemplars.clone(), torch.empty((0, d), device=self.device)


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
        """
        Trả về (acc_history, acc_all, train_elapsed, infer_elapsed).

        acc_history : (T, T) — acc_history[step, i] = acc task i sau step.
        acc_all     : (T,)   — acc từng task sau khi học hết tất cả.

        train_elapsed : chỉ storeFeature + updateMemoryBank (không I/O, không eval).
        infer_elapsed : chỉ final eval forward-pass (sau prefetch).

        Eval giữa chừng bọc bởi _save_rng/_restore_rng → không làm lệch
        random state của luồng train → kết quả reproducible với phiên bản
        không có acc_history nếu cùng seed.
        """
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
                # avg_acc: overall accuracy (weighted) trên tất cả tasks đã thấy
                # Dùng test_idx giống main_v3b — KHÔNG dùng nanmean(acc_history)
                # vì nanmean là unweighted (loãng khi tasks có số mẫu khác nhau)
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
        """
        Trả về (avg_per_trial, last_per_trial, forgetting_per_trial,
                train_time_trials, infer_time_trials) — tất cả shape (N_try,).

        Forgetting theo công thức max (Lopez-Paz 2017):
          Forgetting_i = max_{step=i..T-1}(acc_history[step, i]) − acc_all[i]
        Bỏ task cuối (i = T-1): luôn = 0 → đưa vào mean chỉ làm loãng kết quả.
        """
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
            # acc_all[i] = acc task i sau khi học hết tất cả → mean qua T tasks
            last_per_trial[idx_try] = acc_all.mean().item()

            # ── Forgetting (công thức max, bỏ task cuối) ─────────────────
            # Guard: acc_history chỉ có giá trị khi avg_acc_activ=True vì
            # eval giữa chừng mới được gọi. Nếu False → nan.
            n_tasks = acc_history.shape[0]
            if n_tasks > 1:
                forgetting_list = []
                for i in range(n_tasks - 1):       # bỏ task cuối
                    # acc task i từ step i đến T-1
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
# Runner
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
# Exemplar quality analysis
# =============================================================

def compare_exemplar_diversity(origin_dataset, M=100, metric="cosine",
                               n_classes_analyze=5, n_samples_per_class=500):
    print(f"\n{'='*84}")
    print(f"EXEMPLAR QUALITY ANALYSIS — PER CLASS  (M={M}, metric={metric})")
    print(f"Phân tích {n_classes_analyze} lớp, mỗi lớp tối đa {n_samples_per_class} mẫu")
    print(f"{'='*84}")

    methods = [
        ("Herding",          HerdingSelector,      {}),
        ("Random",           RandomSelector,        {}),
        ("k-DPP α=0",        KDPPSelector,          {"alpha": 0.0}),
        ("k-DPP QD α=1",     KDPPSelector,          {"alpha": 1.0}),
        ("k-DPP QD Offline", KDPPSelectorOffline,   {"alpha": 1.0}),
    ]

    class_feats: dict[int, list] = {}
    for loader in origin_dataset.train_features:
        for inputs, targets in loader:
            for i in range(inputs.size(0)):
                c = int(targets[i])
                if c not in class_feats:
                    class_feats[c] = []
                if len(class_feats[c]) < n_samples_per_class:
                    class_feats[c].append(inputs[i].clone())
        if len(class_feats) >= n_classes_analyze:
            if all(len(v) >= M for v in list(class_feats.values())[:n_classes_analyze]):
                break

    classes_to_use = [c for c in sorted(class_feats.keys())[:n_classes_analyze]
                      if len(class_feats[c]) >= M]
    if not classes_to_use:
        print(f"  Cảnh báo: không đủ mẫu per class cho M={M}, bỏ qua.")
        return {}

    print(f"  Sử dụng {len(classes_to_use)} lớp: {classes_to_use}\n")
    print(f"{'Method':<22} {'MeanErr':>16} {'Spread':>16} {'Coverage':>16} {'OutlierRt':>16}")
    print("-" * 88)

    quality_results = {}
    for name, SelectorClass, sel_kwargs in methods:
        acc_mean_err, acc_spread, acc_coverage, acc_outlier = [], [], [], []
        for c in classes_to_use:
            feats_raw = torch.stack(class_feats[c]).to(device).float()
            sel = SelectorClass(M=M, metric=metric, device=str(device), **sel_kwargs)
            ex_raw, _ = sel.K_DPP(feats_raw)
            if metric == "cosine":
                ex    = F.normalize(ex_raw,    dim=1)
                feats = F.normalize(feats_raw, dim=1)
            else:
                ex, feats = ex_raw, feats_raw
            true_mean = (F.normalize(feats.mean(dim=0).unsqueeze(0), dim=1).squeeze(0)
                         if metric == "cosine" else feats.mean(dim=0))
            ex_mean   = (F.normalize(ex.mean(dim=0).unsqueeze(0), dim=1).squeeze(0)
                         if metric == "cosine" else ex.mean(dim=0))
            mean_err  = (ex_mean - true_mean).norm().item()
            sim_mat   = ex @ ex.T
            idx_u     = torch.triu_indices(ex.shape[0], ex.shape[0], offset=1)
            spread    = ((1.0 - sim_mat[idx_u[0], idx_u[1]]).mean().item()
                         if idx_u[0].numel() > 0 else 0.0)
            sim_to_ex  = feats @ ex.T
            coverage   = (1.0 - sim_to_ex.max(dim=1).values).mean().item()
            dists_all  = (1.0 - (feats @ true_mean)).clamp(min=0)
            dists_ex   = (1.0 - (ex    @ true_mean)).clamp(min=0)
            sigma      = dists_all.std().item() + 1e-8
            outlier_rt = (dists_ex > 1.5 * sigma).float().mean().item() * 100
            acc_mean_err.append(mean_err); acc_spread.append(spread)
            acc_coverage.append(coverage); acc_outlier.append(outlier_rt)

        def ms(lst):
            return round(float(np.mean(lst)), 4), round(float(np.std(lst)), 4)

        me_m, me_s = ms(acc_mean_err); sp_m, sp_s = ms(acc_spread)
        cv_m, cv_s = ms(acc_coverage)
        ol_m = round(float(np.mean(acc_outlier)), 2)
        ol_s = round(float(np.std(acc_outlier)),  2)
        quality_results[name] = {
            "MeanErr_mean": me_m, "MeanErr_std":   me_s,
            "Spread_mean":  sp_m, "Spread_std":    sp_s,
            "Coverage_mean":cv_m, "Coverage_std":  cv_s,
            "OutlierRt_mean":ol_m,"OutlierRt_std": ol_s,
        }
        print(f"{name:<22} {me_m:>7.4f}±{me_s:.4f}  {sp_m:>7.4f}±{sp_s:.4f}  "
              f"{cv_m:>7.4f}±{cv_s:.4f}  {ol_m:>5.2f}%±{ol_s:.2f}%")

    print()
    print("Diễn giải: MeanErr↓  Spread↑  Coverage↓  OutlierRt↓")
    return quality_results


# =============================================================
# Ablation training
# =============================================================

def run_ablation(origin_dataset, memory_budgets, metric="cosine",
                 n_mini_batch=55, N_try=5):
    methods = [
        ("Herding",          HerdingSelector,     {}),
        ("Random",           RandomSelector,       {}),
        ("k-DPP α=0",        KDPPSelector,         {"alpha": 0.0}),
        ("k-DPP QD α=1",     KDPPSelector,         {"alpha": 1.0}),
        ("k-DPP QD Offline", KDPPSelectorOffline,  {"alpha": 1.0}),
    ]

    print(f"\n{'='*104}")
    print(f"ABLATION: Herding vs Random vs k-DPP α=0 vs k-DPP QD α=1 vs k-DPP QD Offline")
    print(f"metric={metric} | N_try={N_try} | n_mini_batch={n_mini_batch}")
    print(f"{'='*104}")
    print(f"{'Method':<22} {'M':>6} | "
          f"{'Avg Acc':>14}  {'Last Acc':>14}  {'Forgetting':>14}  "
          f"{'TrainT':>9}  {'InferT':>9}  {'Time':>7}")
    print("-" * 104)

    results = {}
    for name, SelectorClass, sel_kwargs in methods:
        results[name] = {}
        for m in memory_budgets:
            dataset = origin_dataset.clone()
            (avg_m, avg_s, last_m, last_s, fgt_m, fgt_s,
             tr_m, tr_s, inf_m, inf_s, mem, t) = run_one(
                dataset, SelectorClass, sel_kwargs,
                memorysize=m, metric=metric,
                n_mini_batch=n_mini_batch, N_try=N_try,
            )
            results[name][m] = (avg_m, avg_s, last_m, last_s, fgt_m, fgt_s,
                                 tr_m, tr_s, inf_m, inf_s, mem, t)
            print(f"{name:<22} {m:>6} | "
                  f"{avg_m:>6.2f}±{avg_s:<5.2f}  "
                  f"{last_m:>6.2f}±{last_s:<5.2f}  "
                  f"{fgt_m:>6.2f}±{fgt_s:<5.2f}  "
                  f"{tr_m:>6.2f}±{tr_s:.2f}s  "
                  f"{inf_m:>6.3f}±{inf_s:.3f}s  "
                  f"{t:>6.1f}s")
        print()

    _print_delta_table(results, memory_budgets)
    _print_delta_table(results, memory_budgets,
                       proposed="k-DPP QD Offline",
                       baselines=["Herding", "Random", "k-DPP α=0", "k-DPP QD α=1"])
    return results


def _print_delta_table(results, memory_budgets,
                       proposed="k-DPP QD α=1",
                       baselines=None):
    if baselines is None:
        baselines = ["Herding", "Random", "k-DPP α=0"]
    print(f"\n{'='*96}")
    print(f"DELTA: ({proposed}) − baseline  [dùng mean]")
    print(f"{'Baseline':<22} {'M':>6} | {'ΔAvg':>8} {'ΔLast':>9} {'ΔFgt':>8}  Nhận xét")
    print("-" * 96)
    for baseline_name in baselines:
        if baseline_name not in results or proposed not in results:
            continue
        for m in memory_budgets:
            qd  = results[proposed][m]
            bl  = results[baseline_name][m]
            d_avg  = round(qd[0] - bl[0], 2)
            d_last = round(qd[2] - bl[2], 2)
            d_fgt  = round(qd[4] - bl[4], 2)
            s_a = "▲" if d_avg  > 0 else ("▼" if d_avg  < 0 else "=")
            s_l = "▲" if d_last > 0 else ("▼" if d_last < 0 else "=")
            s_f = "▼" if d_fgt  < 0 else ("▲" if d_fgt  > 0 else "=")
            print(f"{baseline_name:<22} {m:>6} | "
                  f"{s_a}{d_avg:>+6.2f}  {s_l}{d_last:>+7.2f}  {s_f}{d_fgt:>+6.2f}  "
                  f"{_interpret(d_avg, d_last)}")
        print()


def _interpret(d_avg, d_last):
    if d_avg > 1.0 and d_last > 1.0:
        return "Proposed tốt hơn rõ ràng cả hai metric"
    elif d_avg > 1.0:
        return "Proposed tốt Avg, Last tương đương"
    elif d_last > 1.0:
        return "Proposed ít quên hơn (Last tốt hơn rõ)"
    elif d_avg < -1.0 and d_last < -1.0:
        return "Baseline tốt hơn — cần xem lại Proposed"
    elif abs(d_avg) <= 0.5 and abs(d_last) <= 0.5:
        return "Tương đương (±0.5%)"
    else:
        return f"Mixed (avg{d_avg:+.1f}, last{d_last:+.1f})"


# =============================================================
# Excel export
# =============================================================

def _thin():
    s = Side(border_style="thin", color="BDBDBD")
    return Border(left=s, right=s, top=s, bottom=s)

def _hdr(ws, row, col, value, bg="1F4E79", fg="FFFFFF"):
    c = ws.cell(row=row, column=col, value=value)
    c.font      = Font(name="Arial", bold=True, color=fg, size=11)
    c.fill      = PatternFill("solid", fgColor=bg)
    c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    c.border    = _thin()
    return c

def _subhdr(ws, row, col, value):
    return _hdr(ws, row, col, value, bg="2E75B6", fg="FFFFFF")

def _cell(ws, row, col, value, bold=False, center=True, fmt=None, bg=None, fg="000000"):
    c = ws.cell(row=row, column=col, value=value)
    c.font      = Font(name="Arial", bold=bold, color=fg, size=10)
    c.alignment = Alignment(horizontal="center" if center else "left", vertical="center")
    c.border    = _thin()
    if fmt: c.number_format = fmt
    if bg:  c.fill = PatternFill("solid", fgColor=bg)
    return c

def _cell_ms(ws, row, col, mean_val, std_val, is_best=False):
    text = f"{mean_val:.2f} ± {std_val:.2f}"
    fg   = "1F7A4A" if is_best else "000000"
    bg   = "E8F5E9" if is_best else None
    return _cell(ws, row, col, text, bold=is_best, fg=fg, bg=bg)

def _delta(ws, row, col, value, reverse=False):
    good = value < -0.5 if reverse else value > 0.5
    bad  = value > 0.5  if reverse else value < -0.5
    if good:   fg, bg = "1F7A4A", "E8F5E9"
    elif bad:  fg, bg = "B71C1C", "FFEBEE"
    else:      fg, bg = "424242", "F5F5F5"
    c = ws.cell(row=row, column=col, value=value)
    c.font      = Font(name="Arial", bold=True, color=fg, size=10)
    c.fill      = PatternFill("solid", fgColor=bg)
    c.alignment = Alignment(horizontal="center", vertical="center")
    c.number_format = '+0.00;-0.00;"±0"'
    c.border    = _thin()
    return c

def _w(ws, col, width):
    ws.column_dimensions[get_column_letter(col)].width = width


def export_excel(quality_results, ablation_results, memory_budgets,
                 metric, N_try, n_mini_batch, dataset_name, out_path):

    wb  = Workbook()
    wb.remove(wb.active)

    method_order = [
        "Herding", "Random", "k-DPP α=0", "k-DPP QD α=1", "k-DPP QD Offline",
    ]
    row_bg_map = {
        "Herding":          "FFF8F0",
        "Random":           "F8F0FF",
        "k-DPP α=0":       "F0F4FF",
        "k-DPP QD α=1":    "F0FFF4",
        "k-DPP QD Offline": "FFFDE7",
    }

    # ── Sheet 1: Overview ─────────────────────────────────────
    ws0 = wb.create_sheet("Overview")
    ws0.sheet_view.showGridLines = False
    _hdr(ws0, 1, 1,
         "ABLATION STUDY — Exemplar Selection for Online Task-Free CIL",
         bg="1F4E79")
    ws0.merge_cells("A1:C1")
    ws0.row_dimensions[1].height = 30

    cfg = [
        ("Dataset",        dataset_name),
        ("Metric",         metric),
        ("N_try",          N_try),
        ("n_mini_batch",   n_mini_batch),
        ("Memory budgets", ", ".join(str(m) for m in memory_budgets)),
        ("Date/Time",      datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("Device",         str(device)),
    ]
    for i, (k, v) in enumerate(cfg, start=3):
        _cell(ws0, i, 1, k, bold=True,  center=False, bg="DDEEFF")
        _cell(ws0, i, 2, v, bold=False, center=False)
        ws0.merge_cells(f"B{i}:C{i}")

    r = 12
    _hdr(ws0, r, 1, "Giải thích Metrics", bg="37474F")
    ws0.merge_cells(f"A{r}:C{r}")
    ws0.row_dimensions[r].height = 22

    legends = [
        ("Avg Acc (%)",
         "Trung bình accuracy qua các task sau mỗi lần học"),
        ("Last Acc (%)",
         "Mean accuracy từng task sau khi học xong TẤT CẢ task"),
        ("Forgetting (%)",
         "max_{step=i..T-1}(acc[step,i]) − acc_final[i], mean qua T-1 tasks (bỏ task cuối). "
         "Cao = quên nhiều. Lý tưởng ≈ 0"),
        ("Training Time (s)",
         "Thời gian storeFeature + updateMemoryBank thuần (không tính I/O, không tính eval)"),
        ("Inference Time (s)",
         "Thời gian forward-pass final eval (prefetch trước đồng hồ, không tính eval giữa chừng)"),
        ("± std",
         "Độ lệch chuẩn qua N_try trials"),
        ("ΔAvg / ΔLast / ΔFgt",
         "Proposed trừ baseline. Xanh > +0.5%, Đỏ < −0.5%"),
    ]
    for term, desc in legends:
        r += 1
        _cell(ws0, r, 1, term, bold=True, center=False, bg="ECEFF1")
        _cell(ws0, r, 2, desc, center=False)
        ws0.merge_cells(f"B{r}:C{r}")

    _w(ws0, 1, 22); _w(ws0, 2, 90); _w(ws0, 3, 10)

    # ── Sheet 2: Exemplar Quality ─────────────────────────────
    ws1 = wb.create_sheet("Exemplar Quality")
    ws1.sheet_view.showGridLines = False
    _hdr(ws1, 1, 1,
         "EXEMPLAR QUALITY ANALYSIS (Per-Class, mean ± std qua lớp)",
         bg="1F4E79")
    ws1.merge_cells("A1:E1")
    ws1.row_dimensions[1].height = 28
    _cell(ws1, 2, 1, f"metric={metric}", center=False)
    ws1.merge_cells("A2:E2")

    hdrs_q = ["Method", "MeanErr ↓", "Spread ↑", "Coverage ↓", "OutlierRt ↓ (%)"]
    for c, h in enumerate(hdrs_q, 1):
        _subhdr(ws1, 4, c, h)
    ws1.row_dimensions[4].height = 22

    metric_keys = [
        ("MeanErr",   "min"), ("Spread",    "max"),
        ("Coverage",  "min"), ("OutlierRt", "min"),
    ]
    best = {}
    for key, direction in metric_keys:
        vals  = [quality_results.get(n, {}).get(f"{key}_mean", float("nan"))
                 for n in method_order]
        valid = [v for v in vals if not np.isnan(v)]
        if valid:
            best[key] = min(valid) if direction == "min" else max(valid)

    for ri, name in enumerate(method_order, start=5):
        vals  = quality_results.get(name, {})
        bg    = row_bg_map.get(name, "FFFFFF")
        bold  = name in ("k-DPP QD α=1", "k-DPP QD Offline")
        _cell(ws1, ri, 1, name, bold=bold, center=False, bg=bg)
        for ci, (key, _) in enumerate(metric_keys, 2):
            m_val = vals.get(f"{key}_mean", float("nan"))
            s_val = vals.get(f"{key}_std",  float("nan"))
            if np.isnan(m_val):
                _cell(ws1, ri, ci, "N/A", bg=bg)
                continue
            is_best = abs(m_val - best.get(key, float("nan"))) < 1e-6
            _cell_ms(ws1, ri, ci, m_val, s_val, is_best=is_best)

    ws1.freeze_panes = "A5"
    for c, w in [(1,22),(2,18),(3,18),(4,18),(5,20)]:
        _w(ws1, c, w)

    # ── Sheet 3: Ablation Results ─────────────────────────────
    ws2 = wb.create_sheet("Ablation Results")
    ws2.sheet_view.showGridLines = False
    _hdr(ws2, 1, 1,
         "ABLATION: Avg / Last / Forgetting / Train Time / Infer Time (mean ± std)",
         bg="1F4E79")
    ws2.merge_cells("A1:I1")
    ws2.row_dimensions[1].height = 30
    _cell(ws2, 2, 1,
          f"metric={metric} | N_try={N_try} | n_mini_batch={n_mini_batch} | "
          f"Memory: {memory_budgets}",
          center=False)
    ws2.merge_cells("A2:I2")

    hdrs_a = ["Method", "Budget M", "Avg Acc", "Last Acc", "Forgetting",
              "Train Time (s)", "Infer Time (s)", "Mem", "Total Time (s)"]
    for c, h in enumerate(hdrs_a, 1):
        _subhdr(ws2, 3, c, h)
    ws2.row_dimensions[3].height = 30

    r = 4
    for name in method_order:
        is_hl = name in ("k-DPP QD α=1", "k-DPP QD Offline")
        bg    = row_bg_map.get(name, "FFFFFF")
        for m in memory_budgets:
            if m not in ablation_results.get(name, {}):
                continue
            (avg_m, avg_s, last_m, last_s, fgt_m, fgt_s,
             tr_m, tr_s, inf_m, inf_s, mem_cnt, t) = ablation_results[name][m]
            _cell(ws2, r, 1, name,    bold=is_hl, center=False, bg=bg)
            _cell(ws2, r, 2, m,       fmt="#,##0", bg=bg)
            _cell_ms(ws2, r, 3, avg_m,  avg_s)
            _cell_ms(ws2, r, 4, last_m, last_s)
            _cell_ms(ws2, r, 5, fgt_m,  fgt_s)
            _cell_ms(ws2, r, 6, tr_m,   tr_s)
            _cell_ms(ws2, r, 7, inf_m,  inf_s)
            _cell(ws2, r, 8, mem_cnt, fmt="#,##0", bg=bg)
            _cell(ws2, r, 9, t,       fmt="0.00",  bg=bg)
            r += 1
        r += 1

    ws2.freeze_panes = "A4"
    for c, w in [(1,22),(2,10),(3,18),(4,18),(5,18),(6,18),(7,18),(8,8),(9,10)]:
        _w(ws2, c, w)

    # ── Sheet 4: Delta Summary ────────────────────────────────
    ws3 = wb.create_sheet("Delta Summary")
    ws3.sheet_view.showGridLines = False
    _hdr(ws3, 1, 1,
         "DELTA SUMMARY — Proposed methods vs Baselines", bg="1F4E79")
    ws3.merge_cells("A1:H1")
    ws3.row_dimensions[1].height = 28

    hdrs_d = ["Proposed", "Baseline", "Budget M",
              "ΔAvg Acc", "ΔLast Acc", "ΔForgetting", "Nhận xét", "Kết luận"]
    for c, h in enumerate(hdrs_d, 1):
        _subhdr(ws3, 3, c, h)
    ws3.row_dimensions[3].height = 22

    r = 4
    proposed_list = [
        ("k-DPP QD α=1",     ["Herding", "Random", "k-DPP α=0"]),
        ("k-DPP QD Offline", ["Herding", "Random", "k-DPP α=0", "k-DPP QD α=1"]),
    ]
    for proposed_name, baselines in proposed_list:
        for baseline_name in baselines:
            if (proposed_name not in ablation_results or
                    baseline_name not in ablation_results):
                continue
            for m in memory_budgets:
                qd = ablation_results[proposed_name][m]
                bl = ablation_results[baseline_name][m]
                d_avg  = round(qd[0] - bl[0], 2)
                d_last = round(qd[2] - bl[2], 2)
                d_fgt  = round(qd[4] - bl[4], 2)
                interp   = _interpret(d_avg, d_last)
                both_pos = d_avg > 0.5 and d_last > 0.5
                conclude = ("Proposed có đóng góp" if both_pos else
                            "Proposed yếu hơn"     if d_avg < -0.5 and d_last < -0.5 else
                            "Không rõ ràng")
                p_bg = row_bg_map.get(proposed_name,  "FFFFFF")
                b_bg = row_bg_map.get(baseline_name, "FFFFFF")
                _cell(ws3, r, 1, proposed_name,  bold=True, center=False, bg=p_bg)
                _cell(ws3, r, 2, baseline_name,  bold=True, center=False, bg=b_bg)
                _cell(ws3, r, 3, m, fmt="#,##0")
                _delta(ws3, r, 4, d_avg)
                _delta(ws3, r, 5, d_last)
                _delta(ws3, r, 6, d_fgt, reverse=True)
                _cell(ws3, r, 7, interp,   center=False)
                _cell(ws3, r, 8, conclude, bold=True, center=False,
                      bg="D6F5E3" if both_pos else
                         ("FFEBEE" if d_avg < -0.5 else "FFF9C4"))
                r += 1
            r += 1
        r += 1

    ws3.freeze_panes = "A4"
    for c, w in [(1,22),(2,22),(3,10),(4,12),(5,12),(6,14),(7,36),(8,20)]:
        _w(ws3, c, w)

    wb.save(out_path)
    print(f"\n✓ Đã lưu: {out_path}")
    print("  Sheets: Overview | Exemplar Quality | Ablation Results | Delta Summary")


# =============================================================
# Entry point
# =============================================================

if __name__ == '__main__':

    #DATASET_NAME   = "CUB200Step2"
    #origin_dataset = benchmarks.Cub200Resnet50.CUB200RESNET50(start=2, step=2)

    origin_dataset = benchmarks.Core50Resnet18_Claude.CORE50RESNET18()
    DATASET_NAME = "CORe50"

    # origin_dataset = benchmarks.Cifar100Resnet50.CIFAR100RESNET50(start=2, step=2)
    # DATASET_NAME = "CIFAR100Step2"

    MEMORY_BUDGETS = [1000, 3000, 5000]
    METRIC         = "cosine"
    N_TRY          = 5
    N_MINI_BATCH   = 55
    OUTPUT_PATH    = DATASET_NAME + "_Abblation_QD_NCM_Co_the_publish_ExactForgetting_New020626.xlsx"

    quality_results = compare_exemplar_diversity(
        origin_dataset,
        M                   = 100,
        metric              = METRIC,
        n_classes_analyze   = 5,
        n_samples_per_class = 500,
    )

    ablation_results = run_ablation(
        origin_dataset,
        memory_budgets = MEMORY_BUDGETS,
        metric         = METRIC,
        n_mini_batch   = N_MINI_BATCH,
        N_try          = N_TRY,
    )

    export_excel(
        quality_results  = quality_results,
        ablation_results = ablation_results,
        memory_budgets   = MEMORY_BUDGETS,
        metric           = METRIC,
        N_try            = N_TRY,
        n_mini_batch     = N_MINI_BATCH,
        dataset_name     = DATASET_NAME,
        out_path         = OUTPUT_PATH,
    )
