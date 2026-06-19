# mask_generator.py
# المولد الهجين للقناع (Hybrid Mask Generator)
# مُصحَّح للتوافق مع OpenCV 4.x

import numpy as np
import cv2
from typing import Tuple, Optional, List
import time

try:
    from numba import njit, prange
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    prange = range


# ============================================================
# 1. محرك MSER المزدوج (فاتح + غامق) - مُصحَّح للـ OpenCV 4.x
# ============================================================
def mser_dual_extraction(
    image: np.ndarray,
    delta: int = 5,
    min_area: int = 60,
    max_area: int = 14400,
    max_variation: float = 0.25,
    min_diversity: float = 0.2
) -> np.ndarray:
    """
    تطبيق MSER على صورتين: الأصلية والمعكوسة للحصول على مناطق داكنة وفاتحة.
    تم تصحيح أسماء المعاملات لتتوافق مع OpenCV 4.x:
        delta (بدلاً من _delta)
        min_area (بدلاً من _min_area)
        max_area (بدلاً من _max_area)
        max_variation (بدلاً من _max_variation)
        min_diversity (بدلاً من _min_diversity)
    """
    if image.dtype != np.uint8:
        img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        img_uint8 = image.copy()
    
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

    # استخدام الأسماء الصحيحة للمعاملات (بدون شرطات سفلية)
    mser = cv2.MSER_create(
        delta=delta,
        min_area=min_area,
        max_area=max_area,
        max_variation=max_variation,
        min_diversity=min_diversity
    )

    mask = np.zeros_like(gray, dtype=np.uint8)

    # MSER على الصورة الأصلية (مناطق داكنة)
    regions, _ = mser.detectRegions(gray)
    for region in regions:
        pts = region.reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 1)

    # MSER على الصورة المعكوسة (مناطق فاتحة)
    gray_inv = 255 - gray
    regions_inv, _ = mser.detectRegions(gray_inv)
    for region in regions_inv:
        pts = region.reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 1)

    return mask


# ============================================================
# 2. حقل التدرج وزوايا الحواف
# ============================================================
@njit(cache=True)
def compute_gradient_field(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    حساب حقل التدرج وزواياه باستخدام مشتقات صهير.
    """
    H, W = gray.shape
    Gx = np.zeros((H, W), dtype=np.float32)
    Gy = np.zeros((H, W), dtype=np.float32)
    angle = np.zeros((H, W), dtype=np.float32)

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            gx = (gray[y, x+1] - gray[y, x-1]) * 0.5
            gy = (gray[y+1, x] - gray[y-1, x]) * 0.5
            Gx[y, x] = gx
            Gy[y, x] = gy
            angle[y, x] = np.arctan2(gy, gx)

    return Gx, Gy, angle


# ============================================================
# 3. SWT بالأشعة المخروطية (مع تخصيص مسبق للذاكرة)
# ============================================================
@njit(cache=True, parallel=True)
def swt_cone_ray(
    gray: np.ndarray,
    Gx: np.ndarray,
    Gy: np.ndarray,
    edge_mask: np.ndarray,
    path_buffers_y: np.ndarray,
    path_buffers_x: np.ndarray,
    best_path_buffers_y: np.ndarray,
    best_path_buffers_x: np.ndarray,
    cone_angle: float = 0.61,
    max_ray_length: int = 300
) -> np.ndarray:
    """
    قذف أشعة مخروطية مع تخصيص مسبق للذاكرة.
    """
    H, W = gray.shape
    swt_map = np.zeros((H, W), dtype=np.float32)
    angles = np.array([0.0, cone_angle, -cone_angle], dtype=np.float32)

    for y in prange(H):
        path_x = path_buffers_x[y]
        path_y = path_buffers_y[y]
        best_path_x = best_path_buffers_x[y]
        best_path_y = best_path_buffers_y[y]

        for x in range(W):
            if edge_mask[y, x] == 0:
                continue

            gx = Gx[y, x]
            gy = Gy[y, x]
            norm = np.sqrt(gx*gx + gy*gy)
            if norm < 1e-8:
                continue

            dx = gx / norm
            dy = gy / norm

            best_ray_length = 0.0
            best_path_len = 0

            for angle_idx in range(angles.shape[0]):
                offset_angle = angles[angle_idx]
                cos_a = np.cos(offset_angle)
                sin_a = np.sin(offset_angle)
                dir_x = dx * cos_a - dy * sin_a
                dir_y = dy * cos_a + dx * sin_a

                cur_y = float(y)
                cur_x = float(x)
                ray_length = 0.0
                path_len = 0
                hit = False

                step = 0.5
                for step_idx in range(max_ray_length):
                    cur_y += dir_y * step
                    cur_x += dir_x * step
                    ray_length += step

                    yi = int(np.round(cur_y))
                    xi = int(np.round(cur_x))

                    if yi < 0 or yi >= H or xi < 0 or xi >= W:
                        break

                    if path_len < max_ray_length:
                        path_y[path_len] = yi
                        path_x[path_len] = xi
                        path_len += 1

                    if edge_mask[yi, xi] == 1:
                        gx2 = Gx[yi, xi]
                        gy2 = Gy[yi, xi]
                        norm2 = np.sqrt(gx2*gx2 + gy2*gy2)
                        if norm2 > 1e-8:
                            dot = (gx * gx2 + gy * gy2) / (norm * norm2)
                            if dot < -0.3:
                                hit = True
                                break

                if hit:
                    if best_ray_length == 0.0 or ray_length < best_ray_length:
                        best_ray_length = ray_length
                        best_path_len = path_len
                        for p in range(path_len):
                            best_path_x[p] = path_x[p]
                            best_path_y[p] = path_y[p]

            if best_path_len > 0:
                for p in range(best_path_len):
                    py = best_path_y[p]
                    px = best_path_x[p]
                    if swt_map[py, px] == 0.0 or best_ray_length < swt_map[py, px]:
                        swt_map[py, px] = best_ray_length

    return swt_map


# ============================================================
# 4. الفلاتر الإحصائية السبعة
# ============================================================
def apply_statistical_filters(
    swt_map: np.ndarray,
    gray: np.ndarray,
    min_stroke_ratio: float = 0.45,
    min_aspect_ratio: float = 0.1,
    max_aspect_ratio: float = 10.0,
    min_solidity: float = 0.15,
    max_stroke_relative: float = 0.05,
    min_area: int = 20
) -> np.ndarray:
    """
    تطبيق الفلاتر الإحصائية السبعة على المكونات المتصلة المستخرجة من SWT.
    """
    mask_swt = (swt_map > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_swt, connectivity=8
    )

    filtered_mask = np.zeros_like(mask_swt, dtype=np.uint8)

    H, W = swt_map.shape
    max_dim = max(H, W)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        if area < min_area:
            continue

        component_mask = (labels == i)
        swt_values = swt_map[component_mask]
        swt_values = swt_values[swt_values > 0]
        
        if len(swt_values) < 5:
            continue

        mu_swt = np.mean(swt_values)
        sigma_swt = np.std(swt_values)
        
        if mu_swt <= 0:
            continue

        stroke_variation = sigma_swt / mu_swt
        if stroke_variation > (1 - min_stroke_ratio):
            continue

        aspect = width / height if height > 0 else 0
        if aspect < min_aspect_ratio or aspect > max_aspect_ratio:
            continue

        bbox_area = width * height
        solidity = area / bbox_area if bbox_area > 0 else 0
        if solidity < min_solidity:
            continue

        if mu_swt > max_dim * max_stroke_relative:
            continue

        filtered_mask[component_mask] = 1

    return filtered_mask


# ============================================================
# 5. العمليات المورفولوجية
# ============================================================
def morphological_close(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1
) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def morphological_open(
    mask: np.ndarray,
    kernel_size: int = 2,
    iterations: int = 1
) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)


# ============================================================
# 6. المولد الهجين الشامل
# ============================================================
def generate_hybrid_mask(
    image: np.ndarray,
    mser_delta: int = 5,
    mser_min_area: int = 60,
    mser_max_area: int = 14400,
    cone_angle_deg: float = 35.0,
    min_stroke_ratio: float = 0.45,
    min_aspect_ratio: float = 0.1,
    max_aspect_ratio: float = 10.0,
    min_solidity: float = 0.15,
    max_stroke_relative: float = 0.05,
    close_kernel: int = 3,
    close_iterations: int = 1,
    open_kernel: int = 2,
    open_iterations: int = 1,
    max_ray_length: int = 300,
    verbose: bool = True
) -> np.ndarray:
    """
    توليد القناع الهجين باستخدام MSER + SWT + فلاتر إحصائية.
    """
    start = time.perf_counter()

    if image.dtype != np.float32:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = image

    if verbose:
        print("[MaskGen] تشغيل MSER المزدوج...")
    
    mser_mask = mser_dual_extraction(
        img_float,
        delta=mser_delta,
        min_area=mser_min_area,
        max_area=mser_max_area
    )

    if verbose:
        print("[MaskGen] حساب حقل التدرج...")
    
    gray_uint8 = (img_float * 255).astype(np.uint8)
    gray = cv2.cvtColor(gray_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    Gx, Gy, _ = compute_gradient_field(gray)

    if verbose:
        print("[MaskGen] استخراج الحواف المرشحة...")
    
    canny = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
    edge_mask = np.logical_or(mser_mask, canny).astype(np.uint8)

    if verbose:
        print(f"[MaskGen] تنفيذ SWT بالأشعة المخروطية (زاوية {cone_angle_deg:.1f}°)...")
    
    H, W = gray.shape
    path_buffers_y = np.zeros((H, max_ray_length), dtype=np.int32)
    path_buffers_x = np.zeros((H, max_ray_length), dtype=np.int32)
    best_path_buffers_y = np.zeros((H, max_ray_length), dtype=np.int32)
    best_path_buffers_x = np.zeros((H, max_ray_length), dtype=np.int32)

    cone_angle_rad = np.radians(cone_angle_deg)
    swt_map = swt_cone_ray(
        gray, Gx, Gy, edge_mask,
        path_buffers_y, path_buffers_x,
        best_path_buffers_y, best_path_buffers_x,
        cone_angle_rad, max_ray_length
    )

    if verbose:
        print("[MaskGen] تطبيق الفلاتر الإحصائية...")
    
    filtered = apply_statistical_filters(
        swt_map, gray,
        min_stroke_ratio=min_stroke_ratio,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
        min_solidity=min_solidity,
        max_stroke_relative=max_stroke_relative
    )

    if verbose:
        print("[MaskGen] الفتح المورفولوجي...")
    opened = morphological_open(filtered, kernel_size=open_kernel, iterations=open_iterations)

    if verbose:
        print("[MaskGen] الإغلاق المورفولوجي...")
    final_mask = morphological_close(opened, kernel_size=close_kernel, iterations=close_iterations)

    elapsed = time.perf_counter() - start
    if verbose:
        active_pixels = np.sum(final_mask)
        total_pixels = final_mask.shape[0] * final_mask.shape[1]
        coverage = (active_pixels / total_pixels) * 100
        print(f"[MaskGen] اكتمل في {elapsed:.2f} ثانية. بكسلات نشطة: {active_pixels:,} ({coverage:.2f}%)")

    return final_mask.astype(np.uint8)


# ============================================================
# 7. دوال مساعدة
# ============================================================
def visualize_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.3,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    overlay = img_uint8.copy()
    overlay[mask == 1] = color
    return cv2.addWeighted(img_uint8, 1 - alpha, overlay, alpha, 0)


def save_mask(mask: np.ndarray, output_path: str) -> None:
    cv2.imwrite(output_path, (mask * 255).astype(np.uint8))


# ============================================================
# اختبار سريع
# ============================================================
if __name__ == "__main__":
    H, W = 512, 512
    img = np.random.rand(H, W, 3).astype(np.float32)
    img[200:250, 200:300, :] = 0.1
    mask = generate_hybrid_mask(img, verbose=True)
    print(f"\n✅ قناع بأبعاد {mask.shape}, نشط: {np.sum(mask):,}")
