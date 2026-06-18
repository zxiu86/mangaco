"""
engine.py - المحرك السيادي (Orchestrator) المُعاد هيكلته
مع تغذية راجعة ديناميكية للصورة الهدف ومعالجة صحيحة للحواف
"""

import numpy as np
from typing import List, Tuple, Optional
import time

from core import build_gaussian_pyramid, upsample_nnf, initialize_nnf
from propagation import run_patchmatch_iteration, compute_patch_cost_5d

try:
    from numba import njit, prange
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    prange = range


# ============================================================
# 1. إعادة بناء الصورة (من المصدر باستخدام NNF)
# ============================================================
@njit(cache=True, parallel=True)
def _reconstruct_from_source(
    img_source: np.ndarray,
    nnf: np.ndarray,
    mask: np.ndarray,
    output: np.ndarray
) -> np.ndarray:
    """
    تبني صورة هدف نظيفة عن طريق جلب البكسلات من img_source
    باستخدام الإحداثيات المخزنة في NNF.
    """
    H, W, C = img_source.shape

    for y in prange(H):
        for x in range(W):
            if mask[y, x] == 0:
                # الخلفية السليمة تنتقل كما هي
                for c in range(C):
                    output[y, x, c] = img_source[y, x, c]
            else:
                # استخراج المتجه
                x_t = nnf[y, x, 0]
                y_t = nnf[y, x, 1]
                alpha = nnf[y, x, 4]

                # حماية الحدود
                x_t = max(0.0, min(float(W - 1), x_t))
                y_t = max(0.0, min(float(H - 1), y_t))

                xi = int(np.floor(x_t + 0.5))
                yi = int(np.floor(y_t + 0.5))

                for c in range(C):
                    output[y, x, c] = img_source[yi, xi, c] * alpha
    return output


def build_target_from_nnf(
    img_source: np.ndarray,
    nnf: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """بناء صورة هدف نظيفة من المصدر باستخدام NNF."""
    H, W, C = img_source.shape
    output = np.zeros((H, W, C), dtype=np.float32)
    return _reconstruct_from_source(
        np.ascontiguousarray(img_source),
        np.ascontiguousarray(nnf),
        np.ascontiguousarray(mask),
        output
    )


# ============================================================
# 2. حساب التكاليف الأولية (باستخدام مصدر وهدف منفصلين)
# ============================================================
def _compute_initial_costs(
    img_src: np.ndarray,
    img_tgt: np.ndarray,
    mask: np.ndarray,
    nnf: np.ndarray,
    patch_radius: int
) -> np.ndarray:
    H, W = mask.shape
    costs = np.full((H, W), 1e9, dtype=np.float32)
    active_y, active_x = np.where(mask == 1)

    for i in range(len(active_y)):
        y, x = active_y[i], active_x[i]
        costs[y, x] = compute_patch_cost_5d(
            img_src, img_tgt,
            x, y,
            nnf[y, x, 0], nnf[y, x, 1],
            nnf[y, x, 2], nnf[y, x, 3], nnf[y, x, 4],
            patch_radius
        )
    return costs


# ============================================================
# 3. المحرك السيادي الهرمي (مع تحديث ديناميكي للهدف)
# ============================================================
def patchmatch_5d_hierarchical(
    image: np.ndarray,
    mask: np.ndarray,
    levels: int = 4,
    iters_per_level: Optional[List[int]] = None,
    patch_radius: int = 2,
    early_stop_threshold: float = 1e-4,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    محرك PatchMatch 5D الهرمي مع تغذية راجعة للصورة الهدف.
    يتم تحديث الصورة الهدف ديناميكياً بعد كل مستوى باستخدام NNF المحسَّن.
    """
    start_time = time.perf_counter()

    image = np.ascontiguousarray(image, dtype=np.float32)
    mask = np.ascontiguousarray(mask, dtype=np.uint8)

    if verbose:
        print(f"[Engine] بناء الهرم بـ {levels} مستويات...")

    # بناء هرم المصدر فقط (الصورة الأصلية والقناع)
    pyramid_src = build_gaussian_pyramid(image, mask, levels)

    # توزيع التكرارات
    if iters_per_level is None:
        iters_per_level = [max(2, 5 - i) for i in range(levels)]
        iters_per_level.reverse()
        if verbose:
            print(f"[Engine] التكرارات التلقائية: {iters_per_level}")

    if len(iters_per_level) != levels:
        raise ValueError("عدد التكرارات يجب أن يساوي عدد المستويات.")

    # --- المستوى الخشن (البداية) ---
    coarse_idx = levels - 1
    img_src_coarse, mask_coarse = pyramid_src[coarse_idx]
    H_c, W_c = img_src_coarse.shape[:2]

    # تهيئة NNF عشوائياً باستخدام الخلفية السليمة من الصورة المصدر
    nnf = initialize_nnf(H_c, W_c, feature_dim=5, dtype=np.float32)
    nnf = _initialize_random_nnf(img_src_coarse, mask_coarse, nnf)
    nnf = np.ascontiguousarray(nnf)

    # بناء صورة هدف أولية للمستوى الخشن (عن طريق تطبيق NNF البدائي)
    img_tgt_coarse = build_target_from_nnf(img_src_coarse, nnf, mask_coarse)
    img_tgt_coarse = np.ascontiguousarray(img_tgt_coarse)

    # حساب التكاليف الأولية
    costs = _compute_initial_costs(
        img_src_coarse, img_tgt_coarse,
        mask_coarse, nnf, patch_radius
    )

    # --- الحلقات الهرمية الصاعدة ---
    for level in range(levels - 1, -1, -1):
        img_src, mask_curr = pyramid_src[level]

        # إعادة تعميد القناع لضمان قيم ثنائية صرفة (حماية الحواف)
        mask_curr = (mask_curr > 0.5).astype(np.uint8)

        H, W = img_src.shape[:2]

        if level < levels - 1:
            if verbose:
                print(f"[Engine] رفع الـ NNF إلى المستوى {level} ({H}x{W})...")

            # رفع NNF من المستوى الأدنى
            nnf = upsample_nnf(nnf, mask_curr)

            # قص الإحداثيات لتتناسب مع حجم المستوى الحالي
            nnf[:, :, 0] = np.clip(nnf[:, :, 0], 0.0, float(W - 1))
            nnf[:, :, 1] = np.clip(nnf[:, :, 1], 0.0, float(H - 1))

            # --- بناء صورة هدف محدثة لهذا المستوى ---
            # نستخدم NNF المرفوع لجلب بكسلات الخلفية من img_src
            img_tgt = build_target_from_nnf(img_src, nnf, mask_curr)
            img_tgt = np.ascontiguousarray(img_tgt)

            # إعادة حساب التكاليف بناءً على الهدف المحدث
            costs = _compute_initial_costs(
                img_src, img_tgt,
                mask_curr, nnf, patch_radius
            )
        else:
            # المستوى الخشن: نأخذ الهدف الذي بنيناه سابقاً
            img_tgt = img_tgt_coarse

        # عدد التكرارات لهذا المستوى
        max_iters = iters_per_level[level]
        initial_search_radius = max(H, W) / 2.0

        if verbose:
            print(f"[Engine] المستوى {level}: {max_iters} تكرارات، R0={initial_search_radius:.1f}")

        prev_avg_cost = np.mean(costs[mask_curr == 1]) if np.any(mask_curr == 1) else 1e9

        # التكرارات داخل المستوى الحالي
        for iter_idx in range(max_iters):
            nnf, costs = run_patchmatch_iteration(
                img_src, img_tgt,
                mask_curr, nnf, costs,
                iter_num=iter_idx,
                initial_search_radius=initial_search_radius,
                patch_radius=patch_radius
            )

            active_costs = costs[mask_curr == 1]
            avg_cost = np.mean(active_costs) if active_costs.size > 0 else 1e9

            if verbose:
                print(f"  - التكرار {iter_idx+1}/{max_iters}، التكلفة: {avg_cost:.4f}")

            if iter_idx > 0:
                rel_change = abs(avg_cost - prev_avg_cost) / (prev_avg_cost + 1e-12)
                if rel_change < early_stop_threshold:
                    if verbose:
                        print(f"    [تقارب] التغيير النسبي {rel_change:.2e} < {early_stop_threshold:.2e}")
                    break
            prev_avg_cost = avg_cost

        # بعد الانتهاء من تكرارات هذا المستوى،
        # نقوم بتحديث الـ NNF (وهو جاهز للرفع في المستوى التالي).
        # ليست هناك حاجة لتحديث img_tgt هنا لأنه سيُبنى من جديد في المستوى الأعلى.

    # --- إعادة البناء النهائية (باستخدام الصورة الأصلية والـ NNF النهائي) ---
    if verbose:
        print("[Engine] إعادة بناء الصورة النهائية...")

    image_final = build_target_from_nnf(image, nnf, mask)

    elapsed = time.perf_counter() - start_time
    if verbose:
        print(f"[Engine] اكتملت المعالجة في {elapsed:.2f} ثانية")

    return image_final, nnf


# ============================================================
# 4. تهيئة عشوائية (محسّنة)
# ============================================================
def _initialize_random_nnf(
    img_source: np.ndarray,
    mask: np.ndarray,
    nnf: np.ndarray,
    seed: Optional[int] = None
) -> np.ndarray:
    """تهيئة عشوائية باستخدام الخلفية السليمة من img_source."""
    if seed is not None:
        np.random.seed(seed)

    H, W, _ = nnf.shape
    valid_y, valid_x = np.where(mask == 0)
    n_valid = len(valid_y)

    if n_valid == 0:
        raise RuntimeError("لا توجد بكسلات خلفية سليمة للتهيئة العشوائية!")

    active_y, active_x = np.where(mask == 1)
    n_active = len(active_y)

    if n_active == 0:
        return nnf

    random_indices = np.random.randint(0, n_valid, size=n_active)
    x_tgt_all = valid_x[random_indices].astype(np.float32)
    y_tgt_all = valid_y[random_indices].astype(np.float32)

    nnf[active_y, active_x, 0] = x_tgt_all
    nnf[active_y, active_x, 1] = y_tgt_all
    nnf[active_y, active_x, 2] = 0.0
    nnf[active_y, active_x, 3] = 1.0
    nnf[active_y, active_x, 4] = 1.0

    return nnf