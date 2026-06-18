"""
roi_utils.py - أدوات استخراج ودمج مناطق الاهتمام (ROI)
"""

import numpy as np
import cv2
from typing import Tuple, Optional


def extract_roi(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: int = 10
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    استخراج منطقة الاهتمام (ROI) من الصورة والماسك بناءً على Bounding Box مع إضافة هامش.
    
    المعاملات:
        image: الصورة الكاملة (H, W, C) float32 [0,1]
        mask: الماسك الثنائي (H, W) uint8 (1 للمنطقة المطلوب إصلاحها)
        bbox: (x_min, y_min, x_max, y_max) إحداثيات المنطقة المراد معالجتها
        padding: هامش إضافي حول المنطقة لتوفير سياق للخوارزمية
    
    المخرجات:
        roi_image: الصورة المقتطعة
        roi_mask: الماسك المقتطع
        adjusted_bbox: الإحداثيات الجديدة بعد إضافة الهامش (للدمج لاحقاً)
    """
    x_min, y_min, x_max, y_max = bbox
    H, W = image.shape[:2]
    
    # إضافة الهامش مع ضمان البقاء ضمن حدود الصورة
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(W, x_max + padding)
    y_max = min(H, y_max + padding)
    
    roi_image = image[y_min:y_max, x_min:x_max].copy()
    roi_mask = mask[y_min:y_max, x_min:x_max].copy()
    
    return roi_image, roi_mask, (x_min, y_min, x_max, y_max)


def merge_roi(
    image: np.ndarray,
    roi_result: np.ndarray,
    bbox: Tuple[int, int, int, int],
    blend_alpha: float = 0.0  # 0 يعني استبدال كامل، يمكن استخدام تجانس للحواف
) -> np.ndarray:
    """
    دمج نتيجة معالجة الـ ROI مرة أخرى في الصورة الأصلية.
    
    المعاملات:
        image: الصورة الكاملة (سيتم تعديلها في المكان)
        roi_result: الصورة الناتجة من المعالجة بنفس أبعاد الـ ROI
        bbox: (x_min, y_min, x_max, y_max) نفس الإحداثيات المستخدمة في الاستخراج
        blend_alpha: معامل المزج (0 = استبدال كامل، 1 = احتفاظ بالأصل)
    
    المخرجات:
        image: الصورة المحدثة (تم التعديل في المكان)
    """
    x_min, y_min, x_max, y_max = bbox
    h = y_max - y_min
    w = x_max - x_min
    
    if roi_result.shape[:2] != (h, w):
        # إعادة تحجيم النتيجة لتطابق الأبعاد (احتياطي)
        roi_result = cv2.resize(roi_result, (w, h), interpolation=cv2.INTER_LINEAR)
    
    if blend_alpha == 0.0:
        image[y_min:y_max, x_min:x_max] = roi_result
    else:
        # مزج تدريجي لتجنب الحواف القاسية
        mask_region = np.ones((h, w, 1), dtype=np.float32) * blend_alpha
        image[y_min:y_max, x_min:x_max] = (
            image[y_min:y_max, x_min:x_max] * (1 - mask_region) +
            roi_result * mask_region
        )
    
    return image


def compute_bbox_from_strokes(
    strokes: list,
    image_shape: Tuple[int, int],
    padding: int = 10
) -> Optional[Tuple[int, int, int, int]]:
    """
    حساب Bounding Box الذي يغطي جميع ضربات الفرشاة.
    
    المعاملات:
        strokes: قائمة بإحداثيات النقاط (x, y) أو قائمة من الخطوط
        image_shape: (H, W) لتقييد الحدود
        padding: هامش إضافي
    
    المخرجات:
        (x_min, y_min, x_max, y_max) أو None إذا كانت القائمة فارغة
    """
    if not strokes:
        return None
    
    # تجميع جميع النقاط
    xs = []
    ys = []
    for stroke in strokes:
        if isinstance(stroke, list):
            # كل stroke عبارة عن قائمة نقاط (x, y)
            for pt in stroke:
                xs.append(pt[0])
                ys.append(pt[1])
        else:
            # stroke مفردة (x, y)
            xs.append(stroke[0])
            ys.append(stroke[1])
    
    if not xs:
        return None
    
    x_min = max(0, min(xs) - padding)
    x_max = min(image_shape[1], max(xs) + padding)
    y_min = max(0, min(ys) - padding)
    y_max = min(image_shape[0], max(ys) + padding)
    
    # التأكد من أن المنطقة ليست صغيرة جداً
    if x_max - x_min < 5 or y_max - y_min < 5:
        # توسيع المنطقة لتكون على الأقل 5x5
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2
        half = max(5, (max(x_max - x_min, y_max - y_min) // 2) + 5)
        x_min = max(0, cx - half)
        x_max = min(image_shape[1], cx + half)
        y_min = max(0, cy - half)
        y_max = min(image_shape[0], cy + half)
    
    return (x_min, y_min, x_max, y_max)