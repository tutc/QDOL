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
    # methods = [
    #     ("Herding",          HerdingSelector,     {}),
    #     ("Random",           RandomSelector,       {}),
    #     ("k-DPP α=0",        KDPPSelector,         {"alpha": 0.0}),
    #     ("k-DPP QD α=1",     KDPPSelector,         {"alpha": 1.0}),
    #     ("k-DPP QD Offline", KDPPSelectorOffline,  {"alpha": 1.0}),
    # ]

    methods = [
        ("k-DPP α=0",        KDPPSelector,         {"alpha": 0.0}),
        ("k-DPP QD α=1",     KDPPSelector,         {"alpha": 1.0}),
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

    return results



# =============================================================
# Entry point
# =============================================================

if __name__ == '__main__':

    DATASET_NAME   = "CUB200Step2"
    origin_dataset = benchmarks.Cub200Resnet50.CUB200RESNET50(start=2, step=2)

    # origin_dataset = benchmarks.Core50Resnet18_Claude.CORE50RESNET18()
    # DATASET_NAME = "CORe50"

    # origin_dataset = benchmarks.Cifar100Resnet50.CIFAR100RESNET50(start=2, step=2)
    # DATASET_NAME = "CIFAR100Step2"

    MEMORY_BUDGETS = [1000, 3000, 5000]
    METRIC         = "cosine"
    N_TRY          = 5
    N_MINI_BATCH   = 55
    OUTPUT_PATH    = DATASET_NAME + "_Abblation_QD_NCM_Co_the_publish_ExactForgetting_New020626.xlsx"


    ablation_results = run_ablation(
        origin_dataset,
        memory_budgets = MEMORY_BUDGETS,
        metric         = METRIC,
        n_mini_batch   = N_MINI_BATCH,
        N_try          = N_TRY,
    )

