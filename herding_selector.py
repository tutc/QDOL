"""
herding_selector.py
====================
iCaRL-style Herding exemplar selector — drop-in replacement cho KDPPSelector.
Herding (iCaRL, Rebuffi et al. 2017)
"""

import torch
import torch.nn.functional as F


class HerdingSelector:
    """
    iCaRL Herding exemplar selector.

    """

    def __init__(
        self,
        M:      int,
        metric: str = "euclidean",
        device: str = "cuda",
        **kwargs,            # tương thích với HerdingSelector(M, metric, device, alpha=...)
    ):
        if M <= 0:
            raise ValueError(f"M must > 0, received M={M}")
        if metric not in ("euclidean", "cosine"):
            raise ValueError(f"metric must be 'euclidean' or 'cosine', receievd '{metric}'")

        self.M      = M
        self.metric = metric
        self.device = device

        # State (dùng cho Case 3 — lưu exemplar hiện tại)
        self.exemplars = None   # (k, d)

    # =========================================================
    # Internal
    # =========================================================

    def _prep_feats(self, feats: torch.Tensor) -> torch.Tensor:

        feats = feats.to(self.device).float()
        if self.metric == "cosine":
            feats = F.normalize(feats, dim=1)
        return feats

    def _herding(self, feats: torch.Tensor) -> torch.Tensor:
        N, d = feats.shape
        k    = min(self.M, N)

        mu          = feats.mean(dim=0)             # (d,) class mean
        running_sum = torch.zeros(d, device=self.device)
        selected    = []
        remaining   = torch.ones(N, dtype=torch.bool, device=self.device)

        for t in range(k):
            # running mean nếu thêm từng mẫu: (running_sum + feats) / (t+1)
            candidate_means = (running_sum.unsqueeze(0) + feats) / (t + 1)  # (N, d)
            dist2           = ((mu.unsqueeze(0) - candidate_means) ** 2).sum(dim=1)  # (N,)

            # Mask đã chọn → không chọn lại
            dist2[~remaining] = float("inf")

            idx = torch.argmin(dist2).item()
            selected.append(idx)
            remaining[idx]  = False
            running_sum    += feats[idx]

        return torch.tensor(selected, dtype=torch.long, device=self.device)

    # =========================================================
    # Public API — mirror KDPPSelector
    # =========================================================

    def reset(self):
        self.exemplars = None

    def K_DPP(self, all_feats: torch.Tensor):
        all_feats = self._prep_feats(all_feats)
        N, d      = all_feats.shape

        # ---------- Case 1 ----------
        if N <= self.M:
            self.exemplars = all_feats.clone()
            non_exemplars  = torch.empty((0, d), device=self.device)
            return all_feats.clone(), non_exemplars

        idxs = self._herding(all_feats)       # (k,)

        mask          = torch.ones(N, dtype=torch.bool, device=self.device)
        mask[idxs]    = False

        self.exemplars = all_feats[idxs].clone()
        non_exemplars  = all_feats[mask]

        return self.exemplars.clone(), non_exemplars


# =============================================================
# Sanity-check (chạy: python herding_selector.py)
# =============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    M, d = 5, 64

    # ----------------------------------------------------------
    # 1. Functional tests — mirror KDPPSelector tests
    # ----------------------------------------------------------
    print("=== Functional tests ===")
    sel = HerdingSelector(M=M, metric="cosine", device=device)

    # Case 1
    feats_small = torch.randn(3, d)
    ex, non_ex  = sel.K_DPP(feats_small)
    print(f"Case 1 | exemplars: {ex.shape}, non_exemplars: {non_ex.shape}")
    assert ex.shape == (3, d) and non_ex.shape == (0, d), "Case 1 failed"

    # Case 2
    sel.reset()
    feats_full = torch.randn(20, d)
    ex, non_ex = sel.K_DPP(feats_full)
    print(f"Case 2 | exemplars: {ex.shape}, non_exemplars: {non_ex.shape}")
    assert ex.shape == (M, d) and non_ex.shape == (20 - M, d), "Case 2 failed"

    # Case 3
    feats_inc    = torch.cat([ex, torch.randn(1, d, device=device)], dim=0)
    ex2, non_ex2 = sel.K_DPP(feats_inc)
    print(f"Case 3 | exemplars: {ex2.shape}, non_exemplars: {non_ex2.shape}")
    assert ex2.shape == (M, d), "Case 3 failed"
    print("✓ Functional tests passed.\n")

    # ----------------------------------------------------------
    # 2. Mean-matching quality
    # ----------------------------------------------------------
    print("=== Mean-matching quality ===")
    torch.manual_seed(0)
    feats_test = torch.randn(100, d)
    true_mean  = feats_test.mean(dim=0)

    sel_h = HerdingSelector(M=10, metric="euclidean", device=device)
    ex_h, _ = sel_h.K_DPP(feats_test)
    herding_mean_err = (ex_h.mean(dim=0) - true_mean).norm().item()

    # Random baseline
    rand_idxs = torch.randperm(100)[:10]
    rand_mean_err = (feats_test[rand_idxs].mean(dim=0) - true_mean).norm().item()

    print(f"Herding mean error : {herding_mean_err:.4f}")
    print(f"Random  mean error : {rand_mean_err:.4f}")


    print("=== Drop-in compatibility ===")
    sel_compat = HerdingSelector(M=8, metric="euclidean", device=device, alpha=1.0)
    feats_c    = torch.randn(30, d)
    ex_c, _    = sel_compat.K_DPP(feats_c)
    assert ex_c.shape == (8, d), "Drop-in test failed"
 

    print("=== Diversity comparison (det of Gram matrix) ===")
    torch.manual_seed(7)
    feats_div  = torch.randn(50, d)

    def gram_det_proxy(feats_sel):

        G    = feats_sel @ feats_sel.T   # (M, M)
        sign, logdet = torch.linalg.slogdet(G)
        return logdet.item() if sign.item() > 0 else float("-inf")

    sel_herd = HerdingSelector(M=8, metric="euclidean", device=device)
    ex_herd, _ = sel_herd.K_DPP(feats_div)

    rand_idx    = torch.randperm(50)[:8]
    ex_rand     = feats_div[rand_idx]

    det_herd = gram_det_proxy(ex_herd)
    det_rand = gram_det_proxy(ex_rand)

    print(f"Herding log-det (Gram): {det_herd:.2f}")
    print(f"Random  log-det (Gram): {det_rand:.2f}")
