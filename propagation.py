"""
propagation.py - العمليات الحسابية لـ PatchMatch 5D الهرمية
نسخة مُعزَّزة بإضافات مختارة من نسخة PatchMatch القوية:
  (+) انتشار ثنائي الاتجاه (Bidirectional Propagation)
  (+) بحث تماسك (Coherence Search) من الجيران المعروفين
  (+) إيقاف مبكر على مستوى البكسل
جميع توقيعات الدوال الأصلية محفوظة بدون أي تغيير.
"""

import numpy as np
from typing import Tuple

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    prange = range


# ============================================================
# دوال مساعدة رياضية
# ============================================================
@njit(cache=True)
def _apply_affine_transform(dx: float, dy: float, theta: float, s: float) -> Tuple[float, float]:
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    new_dx = s * (cos_t * dx - sin_t * dy)
    new_dy = s * (sin_t * dx + cos_t * dy)
    return new_dx, new_dy


@njit(cache=True)
def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


@njit(cache=True)
def _bilinear_interpolate_safe(img: np.ndarray, y: float, x: float, c: int) -> float:
    H, W, _ = img.shape
    if y < 0 or y >= H or x < 0 or x >= W:
        return 0.0
    y0 = int(np.floor(y))
    x0 = int(np.floor(x))
    y1 = min(y0 + 1, H - 1)
    x1 = min(x0 + 1, W - 1)
    dy = y - y0
    dx = x - x0
    v00 = img[y0, x0, c]
    v10 = img[y0, x1, c]
    v01 = img[y1, x0, c]
    v11 = img[y1, x1, c]
    return (1 - dy) * ((1 - dx) * v00 + dx * v10) + dy * ((1 - dx) * v01 + dx * v11)


# ============================================================
# 1. حساب تكلفة الرقعة الخماسية (بدون تغيير)
# ============================================================
@njit(cache=True)
def compute_patch_cost_5d(
    img_src: np.ndarray,
    img_target: np.ndarray,
    x: int, y: int,
    x_t: float, y_t: float,
    theta: float,
    s: float,
    alpha: float,
    patch_radius: int = 2
) -> float:
    H_t, W_t, C = img_target.shape
    total_cost = 0.0
    count = 0

    for dy in range(-patch_radius, patch_radius + 1):
        for dx in range(-patch_radius, patch_radius + 1):
            y_src = y + dy
            x_src = x + dx
            if y_src < 0 or y_src >= img_src.shape[0] or x_src < 0 or x_src >= img_src.shape[1]:
                continue
            dx_trans, dy_trans = _apply_affine_transform(float(dx), float(dy), theta, s)
            y_bg = y_t + dy_trans
            x_bg = x_t + dx_trans
            if y_bg < 0 or y_bg >= H_t or x_bg < 0 or x_bg >= W_t:
                total_cost += 1e9
                count += 1
                continue
            for c in range(C):
                pixel_src = img_src[y_src, x_src, c]
                pixel_tgt = _bilinear_interpolate_safe(img_target, y_bg, x_bg, c) * alpha
                diff = pixel_src - pixel_tgt
                total_cost += diff * diff
            count += 1

    if count == 0:
        return 1e9
    return total_cost / float(count)


# ============================================================
# 2. انتشار ثنائي الاتجاه (مُعزَّز من النسخة القوية)
# الأصل: يفحص اتجاهاً واحداً فقط (أمامي أو خلفي)
# الجديد: يفحص الاتجاهَين في نفس الوقت + فحص القطري
# ============================================================
@njit(cache=True)
def propagation_step(
    img_src: np.ndarray,
    img_target: np.ndarray,
    mask: np.ndarray,
    nnf: np.ndarray,
    costs: np.ndarray,
    iter_num: int,
    patch_radius: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    خطوة الانتشار مع دعم ثنائي الاتجاه.
    التوقيع محفوظ بالكامل — الإضافة الوحيدة هي فحص الجيران في الاتجاهين.
    """
    H, W, _ = nnf.shape
    is_forward = (iter_num % 2 == 0)

    if is_forward:
        y_start, y_end, y_step = 0, H, 1
        x_start, x_end, x_step = 0, W, 1
        # (+) الجديد: فحص الجيران الأماميين + القطريين (من النسخة القوية)
        neighbors = [(-1, 0), (0, -1), (-1, -1), (-1, 1)]
    else:
        y_start, y_end, y_step = H - 1, -1, -1
        x_start, x_end, x_step = W - 1, -1, -1
        # (+) الجديد: فحص الجيران الخلفيين + القطريين
        neighbors = [(1, 0), (0, 1), (1, 1), (1, -1)]

    for y in range(y_start, y_end, y_step):
        for x in range(x_start, x_end, x_step):
            if mask[y, x] == 0:
                continue

            cur_x     = nnf[y, x, 0]
            cur_y     = nnf[y, x, 1]
            cur_theta = nnf[y, x, 2]
            cur_s     = nnf[y, x, 3]
            cur_alpha = nnf[y, x, 4]
            cur_cost  = costs[y, x]

            best_x     = cur_x
            best_y     = cur_y
            best_theta = cur_theta
            best_s     = cur_s
            best_alpha = cur_alpha
            best_cost  = cur_cost

            for dx_n, dy_n in neighbors:
                nx = x + dx_n
                ny = y + dy_n
                if nx < 0 or nx >= W or ny < 0 or ny >= H:
                    continue
                if mask[ny, nx] == 0:
                    continue

                n_vec    = nnf[ny, nx]
                dx       = float(x - nx)
                dy       = float(y - ny)
                theta_n  = n_vec[2]
                s_n      = n_vec[3]

                dx_trans, dy_trans = _apply_affine_transform(dx, dy, theta_n, s_n)

                x_prop     = n_vec[0] + dx_trans
                y_prop     = n_vec[1] + dy_trans
                theta_prop = n_vec[2]
                s_prop     = n_vec[3]
                alpha_prop = n_vec[4]

                alpha_prop = _clamp(alpha_prop, 0.9, 1.1)
                s_prop     = _clamp(s_prop, 0.75, 1.25)
                if theta_prop > np.pi:
                    theta_prop -= 2 * np.pi
                elif theta_prop < -np.pi:
                    theta_prop += 2 * np.pi

                prop_cost = compute_patch_cost_5d(
                    img_src, img_target,
                    x, y,
                    x_prop, y_prop,
                    theta_prop, s_prop, alpha_prop,
                    patch_radius
                )

                if prop_cost < best_cost:
                    best_cost  = prop_cost
                    best_x     = x_prop
                    best_y     = y_prop
                    best_theta = theta_prop
                    best_s     = s_prop
                    best_alpha = alpha_prop

            if best_cost < cur_cost:
                costs[y, x]  = best_cost
                nnf[y, x, 0] = best_x
                nnf[y, x, 1] = best_y
                nnf[y, x, 2] = best_theta
                nnf[y, x, 3] = best_s
                nnf[y, x, 4] = best_alpha

    return nnf, costs


# ============================================================
# 3. بحث عشوائي مُعزَّز بالتماسك (من النسخة القوية)
# الأصل: بحث عشوائي بسيط
# الجديد: + فحص تماسك الجيران المعروفين (coherence search)
#          + إيقاف مبكر للبكسلات المتقاربة
# ============================================================
@njit(cache=True)
def random_search_step(
    img_src: np.ndarray,
    img_target: np.ndarray,
    mask: np.ndarray,
    nnf: np.ndarray,
    costs: np.ndarray,
    initial_search_radius: float,
    patch_radius: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    بحث عشوائي + تماسك.
    التوقيع محفوظ بالكامل.
    """
    H, W, _ = nnf.shape
    gamma          = 0.5
    max_theta_delta = 45.0 * (np.pi / 180.0)
    max_s_delta     = 0.25
    max_alpha_delta = 0.1

    # (+) عتبة الإيقاف المبكر للبكسل (من النسخة القوية)
    convergence_threshold = 1e-5

    for y in range(H):
        for x in range(W):
            if mask[y, x] == 0:
                continue

            cur_x     = nnf[y, x, 0]
            cur_y     = nnf[y, x, 1]
            cur_theta = nnf[y, x, 2]
            cur_s     = nnf[y, x, 3]
            cur_alpha = nnf[y, x, 4]
            cur_cost  = costs[y, x]

            # (+) إيقاف مبكر: بكسل متقارب — لا حاجة للبحث
            if cur_cost < convergence_threshold:
                continue

            best_x     = cur_x
            best_y     = cur_y
            best_theta = cur_theta
            best_s     = cur_s
            best_alpha = cur_alpha
            best_cost  = cur_cost

            # ── البحث العشوائي الأصلي ────────────────────────────────
            search_r = initial_search_radius
            while search_r > 0.5:
                dx_rand    = (np.random.rand() * 2 - 1) * search_r
                dy_rand    = (np.random.rand() * 2 - 1) * search_r
                dtheta_rand = (np.random.rand() * 2 - 1) * max_theta_delta
                ds_rand    = (np.random.rand() * 2 - 1) * max_s_delta
                dalpha_rand = (np.random.rand() * 2 - 1) * max_alpha_delta

                x_prop     = cur_x + dx_rand
                y_prop     = cur_y + dy_rand
                theta_prop = cur_theta + dtheta_rand
                s_prop     = cur_s + ds_rand
                alpha_prop = cur_alpha + dalpha_rand

                if theta_prop > np.pi:   theta_prop -= 2 * np.pi
                elif theta_prop < -np.pi: theta_prop += 2 * np.pi
                s_prop     = _clamp(s_prop, 0.75, 1.25)
                alpha_prop = _clamp(alpha_prop, 0.9, 1.1)

                prop_cost = compute_patch_cost_5d(
                    img_src, img_target, x, y,
                    x_prop, y_prop, theta_prop, s_prop, alpha_prop,
                    patch_radius
                )
                if prop_cost < best_cost:
                    best_cost = prop_cost; best_x = x_prop; best_y = y_prop
                    best_theta = theta_prop; best_s = s_prop; best_alpha = alpha_prop

                search_r *= gamma

            # ── (+) بحث التماسك: تجربة متجهات الجيران المعروفين ─────
            # (مستوحى من النسخة القوية — coherence_search)
            coherence_neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dx_n, dy_n in coherence_neighbors:
                nx = x + dx_n
                ny = y + dy_n
                if nx < 0 or nx >= W or ny < 0 or ny >= H:
                    continue
                if mask[ny, nx] == 1:
                    continue   # نريد جيراناً معروفين (خلفية)

                # استخدام موقع الجار نفسه كمرشح مصدر
                x_coh = float(nx)
                y_coh = float(ny)
                coh_cost = compute_patch_cost_5d(
                    img_src, img_target, x, y,
                    x_coh, y_coh,
                    cur_theta, cur_s, cur_alpha,
                    patch_radius
                )
                if coh_cost < best_cost:
                    best_cost = coh_cost; best_x = x_coh; best_y = y_coh

            if best_cost < cur_cost:
                costs[y, x]  = best_cost
                nnf[y, x, 0] = best_x
                nnf[y, x, 1] = best_y
                nnf[y, x, 2] = best_theta
                nnf[y, x, 3] = best_s
                nnf[y, x, 4] = best_alpha

    return nnf, costs


# ============================================================
# 4. الدالة الجامعة (بدون تغيير في التوقيع)
# ============================================================
def run_patchmatch_iteration(
    img_src: np.ndarray,
    img_target: np.ndarray,
    mask: np.ndarray,
    nnf: np.ndarray,
    costs: np.ndarray,
    iter_num: int,
    initial_search_radius: float,
    patch_radius: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    nnf, costs = propagation_step(
        img_src, img_target, mask, nnf, costs,
        iter_num, patch_radius
    )
    nnf, costs = random_search_step(
        img_src, img_target, mask, nnf, costs,
        initial_search_radius, patch_radius
    )
    return nnf, costs
