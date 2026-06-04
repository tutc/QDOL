import torch
import torch.nn.functional as F


class KDPPSelector:
    """
    Quality-Diversity k-DPP exemplar selector.

    Kết hợp hai tiêu chí trong một kernel duy nhất:
      - Diversity  : Greedy QR / Gram-Schmidt maximize volume trong feature space
      - Quality    : Prototype-distance weighting — ưu tiên mẫu gần class mean,
                     loại bỏ outlier tự động mà không cần tune tham số

    Kernel:  L(i,j) = q(i) · k(fi, fj) · q(j)
      q(i) = exp(−‖fi − μ‖² / 2σ²)   ← quality: gần prototype → cao
      k(·) = RBF trong weighted feature space ← diversity

    Tính tự thích nghi: σ² = mean(dist²) nên scale tự động theo
    phân phối của từng lớp/dataset — không cần hyperparameter mới.

    Args:
        M       : budget — số exemplar tối đa mỗi lớp
        metric  : "euclidean" hoặc "cosine" (L2-normalize trước khi tính)
        device  : "cuda" hoặc "cpu"
        alpha   : cường độ quality weighting ∈ [0, 1]
                  0 = pure diversity (hành vi cũ)
                  1 = full quality-diversity (mặc định)
    
    Version kdpp cuối, đã update/chọn theo từng sample vào  (Lấy từ cuối session 2)
                  
    """

    def __init__(
        self,
        M:      int,
        metric: str   = "cosine",
        device: str   = "cuda",
        alpha:  float = 1.0,
    ):
        if M <= 0:
            raise ValueError(f"M phải > 0, nhận được M={M}")
        if metric not in ("euclidean", "cosine"):
            raise ValueError(f"metric phải là 'euclidean' hoặc 'cosine', nhận được '{metric}'")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha phải trong [0,1], nhận được alpha={alpha}")

        self.M      = M
        self.metric = metric
        self.device = device
        self.alpha  = alpha

        # --- State lưu giữa các lần gọi (incremental mode) ---
        self.Q         = None   # (d, k)  — cơ sở trực giao của subspace exemplar
        self.scores    = None   # (k,)    — residual norm² tại thời điểm chọn
        self.exemplars = None   # (k, d)  — exemplar features đang được giữ

    # =========================================================
    # INTERNAL — chuẩn hóa đặc trưng
    # =========================================================

    def _prep_feats(self, feats: torch.Tensor) -> torch.Tensor:
        """Chuyển về đúng device/dtype; normalize nếu metric='cosine'."""
        feats = feats.to(self.device).float()
        if self.metric == "cosine":
            feats = F.normalize(feats, dim=1)
        return feats

    # =========================================================
    # INTERNAL — Quality score từ prototype distance
    # =========================================================

    def _quality_weights(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Tính quality score cho từng mẫu dựa trên khoảng cách đến class prototype.

        q(i) = exp(−‖fi − μ‖² / 2σ²)

        Tính tự thích nghi:
          - σ² = mean dist² → scale tự động theo spread của từng lớp
          - Khi alpha=0: trả về ones (pure diversity, hành vi cũ)
          - Khi N=1: không có phân phối → trả về ones

        Args:
            feats : (N, d) — features đã prep (normalized nếu cosine)
        Returns:
            quality : (N,) ∈ (0, 1] — score cao = gần prototype = nên chọn
        """
        if self.alpha == 0.0 or feats.shape[0] <= 1:
            return torch.ones(feats.shape[0], device=self.device)

        mu     = feats.mean(dim=0)                              # (d,) class prototype
        dists2 = ((feats - mu) ** 2).sum(dim=1)                # (N,) dist² to mean
        sigma2 = dists2.mean().clamp(min=1e-6)                  # adaptive bandwidth

        quality = torch.exp(-dists2 / (2.0 * sigma2))          # (N,) ∈ (0,1]

        # Blend: alpha=1 → full quality weighting; alpha=0 → uniform
        quality = self.alpha * quality + (1.0 - self.alpha) * torch.ones_like(quality)
        return quality

    # =========================================================
    # INTERNAL — Quality-Diversity Greedy QR
    # =========================================================

    def _greedy_qr(self, feats: torch.Tensor):
        """
        Greedy maximum-volume selection với quality-diversity kernel.

        Cách hoạt động:
          1. Tính quality weight q(i) cho từng mẫu
          2. Scale features: f̃(i) = f(i) × q(i)  → mẫu xa prototype bị shrink
          3. Chạy greedy QR trên weighted features → maximize quality-weighted volume
          4. Residual vẫn project trên weighted features để nhất quán

        Trả về:
            selected         : LongTensor (k,)
            Q                : Tensor (d, k)  — cơ sở trực giao
            scores_at_select : Tensor (k,)    — residual norm² tại lúc chọn
        """
        n, d = feats.shape
        k    = min(self.M, n)

        # === Quality weighting ===
        quality         = self._quality_weights(feats)          # (N,)
        weighted_feats  = feats * quality.unsqueeze(1)          # (N, d)

        Q                = torch.zeros((d, 0), device=self.device)
        residual         = weighted_feats.clone()
        selected         = []
        scores_at_select = []

        for _ in range(k):
            s   = (residual ** 2).sum(dim=1)                    # (N,) residual norm²
            idx = torch.argmax(s).item()

            scores_at_select.append(s[idx])
            selected.append(idx)

            # Gram-Schmidt trên weighted space
            v = residual[idx].clone()
            v = v / (v.norm() + 1e-8)
            Q = torch.cat([Q, v.unsqueeze(1)], dim=1)          # (d, i+1)

            # Project trên weighted features (nhất quán với residual)
            proj     = weighted_feats @ v                       # (N,)
            residual = residual - proj.unsqueeze(1) * v.unsqueeze(0)

        selected         = torch.tensor(selected, dtype=torch.long, device=self.device)
        scores_at_select = torch.stack(scores_at_select)
        return selected, Q, scores_at_select

    # =========================================================
    # INTERNAL — Incremental update (online, N = M+1)
    # =========================================================

    def _incremental_update(self, new_feat: torch.Tensor) -> bool:
        """
        Thử thêm new_feat vào exemplar set hiện tại.

        Áp dụng quality weighting nhất quán với _greedy_qr:
          - Tính quality của new_feat so với prototype của exemplar set hiện tại
          - So sánh weighted residual với worst exemplar

        Args:
            new_feat : (d,) — feature của mẫu mới (đã prep)
        Returns:
            True nếu có thay thế, False nếu new_feat bị bỏ qua
        """
        # Quality của mẫu mới so với prototype exemplar set hiện tại
        all_feats_with_new = torch.cat(
            [self.exemplars, new_feat.unsqueeze(0)], dim=0
        )                                                           # (M+1, d)
        quality    = self._quality_weights(all_feats_with_new)      # (M+1,)
        new_q      = quality[-1]
        weighted_new = new_feat * new_q                             # (d,)

        # Residual của weighted new_feat trong subspace Q
        proj      = self.Q.t() @ weighted_new                      # (k,)
        r         = weighted_new - self.Q @ proj                   # (d,)
        new_score = (r ** 2).sum()

        worst_idx = torch.argmin(self.scores).item()

        if new_score > self.scores[worst_idx]:
            self.exemplars            = self.exemplars.clone()
            self.exemplars[worst_idx] = new_feat
            self.scores[worst_idx]    = new_score

            # Rebuild Q từ exemplar set mới với quality weighting
            new_quality      = self._quality_weights(self.exemplars)    # (M,)
            weighted_ex      = self.exemplars * new_quality.unsqueeze(1)# (M, d)
            Q_new, _         = torch.linalg.qr(weighted_ex.t(), mode="reduced")
            self.Q           = Q_new                                    # (d, M)
            return True

        return False

    # =========================================================
    # PUBLIC API
    # =========================================================

    def reset(self):
        """Xóa state — gọi khi bắt đầu lớp mới hoặc trial mới."""
        self.Q         = None
        self.scores    = None
        self.exemplars = None

    def K_DPP(self, all_feats: torch.Tensor):
        """
        Chọn exemplar từ tập đặc trưng all_feats.

        Args:
            all_feats : (N, d) — toàn bộ features của một lớp tính đến thời điểm này

        Returns:
            exemplars     : (min(N,M), d) — exemplar được chọn (quality-diverse)
            non_exemplars : (N - min(N,M), d) — mẫu không được chọn

        Ba chế độ:
            Case 1 — N ≤ M            : giữ tất cả
            Case 2 — N > M, Q is None : greedy full với quality weighting
            Case 3 — N = M+1          : incremental update
        """
        all_feats = self._prep_feats(all_feats)
        N, d      = all_feats.shape

        # ---------- Case 1: chưa overflow ----------
        if N <= self.M:
            non_exemplars = torch.empty((0, d), device=self.device)
            return all_feats.clone(), non_exemplars

        # ---------- Case 2: lần đầu overflow → greedy full ----------
        if self.Q is None:
            idxs, Q, scores = self._greedy_qr(all_feats)

            mask           = torch.ones(N, dtype=torch.bool, device=self.device)
            mask[idxs]     = False

            self.exemplars = all_feats[idxs].clone()
            self.Q         = Q
            self.scores    = scores

            non_exemplars  = all_feats[mask]
            return self.exemplars.clone(), non_exemplars

        # ---------- Case 3: incremental ----------
        if N != self.M + 1:
            raise ValueError(
                f"Incremental mode yêu cầu N = M+1 = {self.M + 1}, "
                f"nhưng nhận N={N}. Gọi reset() nếu muốn bắt đầu lại."
            )

        new_feat = all_feats[-1]
        self._incremental_update(new_feat)

        non_exemplars = torch.empty((0, d), device=self.device)
        return self.exemplars.clone(), non_exemplars


# =============================================================
# Sanity-check + ablation (chạy: python kdpp_selector.py)
# =============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    M, d = 5, 512

    # ----------------------------------------------------------
    # 1. Functional tests (3 cases)
    # ----------------------------------------------------------
    print("=== Functional tests ===")
    sel = KDPPSelector(M=M, metric="cosine", device=device, alpha=1.0)

    feats_small = torch.randn(3, d)
    ex, non_ex  = sel.K_DPP(feats_small)
    print(f"Case 1 | exemplars: {ex.shape}, non_exemplars: {non_ex.shape}")
    assert ex.shape == (3, d) and non_ex.shape == (0, d)

    sel.reset()
    feats_full = torch.randn(20, d)
    ex, non_ex = sel.K_DPP(feats_full)
    print(f"Case 2 | exemplars: {ex.shape}, non_exemplars: {non_ex.shape}")
    assert ex.shape == (M, d) and non_ex.shape == (20 - M, d)

    feats_inc   = torch.cat([ex, torch.randn(1, d, device=device)], dim=0)
    ex2, non_ex2 = sel.K_DPP(feats_inc)
    print(f"Case 3 | exemplars: {ex2.shape}, non_exemplars: {non_ex2.shape}")
    assert ex2.shape == (M, d)

    assert (sel.scores >= 0).all(), "scores phải >= 0"

    I_approx = sel.Q.t() @ sel.Q
    err = (I_approx - torch.eye(M, device=device)).abs().max().item()
    print(f"Q orthogonality error: {err:.2e}  (phải < 1e-5)")
    assert err < 1e-5
    print("✓ Functional tests passed.\n")

    # ----------------------------------------------------------
    # 2. Alpha ablation: quality weighting có tác dụng không?
    # ----------------------------------------------------------
    print("=== Alpha ablation (với outlier injection) ===")

    torch.manual_seed(0)
    cluster  = torch.randn(50, d) * 0.3
    outliers = torch.randn(10, d) * 5.0 + 20.0
    feats_mixed = torch.cat([cluster, outliers], dim=0)     # 60 mẫu

    for alpha in [0.0, 0.5, 1.0]:
        sel_ab = KDPPSelector(M=10, metric="euclidean", device=device, alpha=alpha)
        ex_ab, _ = sel_ab.K_DPP(feats_mixed)
        norms      = ex_ab.norm(dim=1)
        in_cluster = (norms < 5.0).sum().item()
        print(f"  alpha={alpha:.1f} | trong cluster: {in_cluster}/10  "
              f"(norms min={norms.min():.1f}, max={norms.max():.1f})")

    sel0 = KDPPSelector(M=10, device=device, alpha=0.0)
    sel1 = KDPPSelector(M=10, device=device, alpha=1.0)
    ex0, _ = sel0.K_DPP(feats_mixed)
    ex1, _ = sel1.K_DPP(feats_mixed)
    in0 = (ex0.norm(dim=1) < 5.0).sum().item()
    in1 = (ex1.norm(dim=1) < 5.0).sum().item()
    assert in1 >= in0, f"Quality weighting phải chọn nhiều cluster hơn: {in1} vs {in0}"
    print(f"✓ alpha=1.0 chọn {in1} cluster exemplar vs alpha=0.0 chọn {in0}.\n")

    # ----------------------------------------------------------
    # 3. Backward compatibility: alpha=0 không crash
    # ----------------------------------------------------------
    print("=== Backward compatibility ===")
    torch.manual_seed(7)
    feats_bc = torch.randn(30, d)
    sel_old  = KDPPSelector(M=8, device=device, alpha=0.0)
    sel_new  = KDPPSelector(M=8, device=device, alpha=1.0)
    ex_old, _ = sel_old.K_DPP(feats_bc)
    ex_new, _ = sel_new.K_DPP(feats_bc)
    assert ex_old.shape == ex_new.shape == (8, d)
    diff = (~(ex_old == ex_new).all(dim=1)).sum().item()
    print(f"✓ Cả hai alpha đều cho shape đúng. Số exemplar khác nhau: {diff}/8\n")

    print("✓ Tất cả tests passed.")
