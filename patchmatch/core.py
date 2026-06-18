"""
core.py - البنى الأساسية وإدارة الذاكرة لنظام PatchMatch 5D الهرمي
(نسخة فائقة الأداء - مع تحسينات الذاكرة والسرعة)
"""

import numpy as np
from typing import List, Tuple, Optional
import warnings

try:
    import cv2
except ImportError:
    cv2 = None
    warnings.warn("OpenCV (cv2) غير مثبت. الأداء سيكون دون المستوى.", ImportWarning)

# ------------------------------------------------------------
# 1. بنية البيانات الأساسية
# ------------------------------------------------------------
def initialize_nnf(height: int, width: int, feature_dim: int = 5, dtype: np.dtype = np.float32) -> np.ndarray:
    return np.ascontiguousarray(np.zeros((height, width, feature_dim), dtype=dtype))

def allocate_compressed_nnf(height: int, width: int) -> np.ndarray:
    dtype = np.dtype([
        ('xy', np.int16, (2,)),  # xy[0] = المحور X (الأفقي), xy[1] = المحور Y (العمودي)
        ('theta', np.uint8),
        ('s', np.float16),
        ('alpha', np.float16)
    ])
    return np.ascontiguousarray(np.zeros((height, width), dtype=dtype))

# ------------------------------------------------------------
# 2. دوال التكميم (مع تحسين الأداء المتجهي)
# ------------------------------------------------------------
def quantize_angle(theta_deg: float) -> int:
    theta_clipped = theta_deg % 360.0
    # استخدام صيغة مضغوطة: الضرب بثابت ثم التقريب بإضافة 0.5 وتحويل int (أسرع من np.round)
    computed = theta_clipped * (255.0 / 360.0) + 0.5
    return int(computed) % 256

def dequantize_angle(theta_quantized: int) -> float:
    return (theta_quantized / 255.0) * 360.0

def pack_nnf_to_compressed(nnf_float: np.ndarray) -> np.ndarray:
    """
    تحويل NNF إلى بنية مضغوطة مع تحسين أداء التكميم المتجهي.
    [التحسين]: استخدام عمليات مصفوفية مباشرة وتجنب np.round.
    """
    H, W, _ = nnf_float.shape
    compressed = allocate_compressed_nnf(H, W)
    
    # الإحداثيات المكانية (قص آمن)
    xy = np.clip(nnf_float[:, :, 0:2], -32768, 32767).astype(np.int16)
    compressed['xy'] = xy
    
    # ------------------------------------------------------------
    # التحسين رقم 3: تكميم الزوايا بطريقة متجهية فائقة السرعة
    # بدلاً من القسمة والضرب المنفصلين، نضرب في ثابت واحد (255.0/360.0)
    # وإضافة 0.5 ثم التحويل إلى int يحل محل np.round (أسرع بكثير)
    # ------------------------------------------------------------
    theta_deg = nnf_float[:, :, 2] % 360.0
    theta_quantized = theta_deg * (255.0 / 360.0) + 0.5
    compressed['theta'] = theta_quantized.astype(np.uint8)  # التحويل يقطع الكسور تلقائياً
    
    # المقياس والإضاءة
    compressed['s'] = nnf_float[:, :, 3].astype(np.float16)
    compressed['alpha'] = nnf_float[:, :, 4].astype(np.float16)
    
    return np.ascontiguousarray(compressed)

def unpack_compressed_to_float(compressed_nnf: np.ndarray) -> np.ndarray:
    H, W = compressed_nnf.shape
    nnf_float = np.zeros((H, W, 5), dtype=np.float32)
    nnf_float[:, :, 0:2] = compressed_nnf['xy'].astype(np.float32)
    nnf_float[:, :, 2] = dequantize_angle(compressed_nnf['theta'].astype(np.float32))
    nnf_float[:, :, 3] = compressed_nnf['s'].astype(np.float32)
    nnf_float[:, :, 4] = compressed_nnf['alpha'].astype(np.float32)
    return np.ascontiguousarray(nnf_float)

# ------------------------------------------------------------
# 3. بناء الهرم (مع تحسين استهلاك الذاكرة)
# ------------------------------------------------------------
def build_gaussian_pyramid(image: np.ndarray, mask: np.ndarray, levels: int = 4) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    بناء الهرم الغاوسي مع تحسين الذاكرة عبر إعادة استخدام المخازن المؤقتة.
    [التحسين رقم 2]: استخدام وسيط dst لتقليل التخصيصات المؤقتة.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV مطلوب لبناء الهرم.")
    
    pyramid = []
    current_img = image.copy()
    current_mask = mask.astype(np.float32) / 255.0 if mask.dtype != np.float32 else mask
    
    for level in range(levels):
        if current_img.shape[:2] != current_mask.shape[:2]:
            raise ValueError("تطابق أبعاد الصورة والقناع مطلوب.")
        
        binary_mask = (current_mask > 0.15).astype(np.uint8)
        pyramid.append((current_img.copy(), binary_mask))
        
        if level < levels - 1:
            new_h = current_img.shape[0] // 2
            new_w = current_img.shape[1] // 2
            
            # ------------------------------------------------------------
            # التحسين: استخدام dst لتعديل المصفوفة في مكانها (In-place)
            # نقوم بتخصيص مصفوفة جديدة للصورة المصغرة، لكننا نتجنب الاحتفاظ بالنسخة القديمة
            # عبر إعادة استخدام متغير current_img نفسه باستخدام dst.
            # ------------------------------------------------------------
            resized_img = np.empty((new_h, new_w, current_img.shape[2]), dtype=current_img.dtype)
            cv2.resize(current_img, (new_w, new_h), dst=resized_img, interpolation=cv2.INTER_AREA)
            current_img = resized_img  # الآن replaced، ولا توجد نسخة إضافية في الذاكرة
            
            # نفس الشيء للقناع
            resized_mask = np.empty((new_h, new_w), dtype=current_mask.dtype)
            cv2.resize(current_mask, (new_w, new_h), dst=resized_mask, interpolation=cv2.INTER_LINEAR)
            current_mask = resized_mask
    
    return pyramid

# ------------------------------------------------------------
# 4. استخراج الصناديق (Bounding Boxes)
# ------------------------------------------------------------
def extract_bounding_boxes(mask: np.ndarray, min_area: int = 100) -> List[Tuple[int, int, int, int]]:
    if cv2 is None:
        raise RuntimeError("OpenCV مطلوب لاستخراج الصناديق.")
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)  # x: عرض, y: ارتفاع
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(mask.shape[1] - x, w + 2 * padding)
        h = min(mask.shape[0] - y, h + 2 * padding)
        bboxes.append((x, y, w, h))
    return bboxes

# ------------------------------------------------------------
# 5. رفع مستوى NNF (مع تحسينات الذاكرة والسرعة)
# ------------------------------------------------------------
def upsample_nnf(nnf_low: np.ndarray, mask_high: Optional[np.ndarray] = None) -> np.ndarray:
    """
    رفع مستوى حقل المتجهات مع عزل المناطق النصية (Mask-Driven).
    [التحسين رقم 1]: استخدام وسيط dst لتجنب التخصيصات المتكررة داخل الحلقات.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV مطلوب لرفع المستوى.")
    
    H_low, W_low, _ = nnf_low.shape
    H_high = H_low * 2
    W_high = W_low * 2
    
    scale_x = W_high / W_low
    scale_y = H_high / H_low
    
    nnf_upsampled = np.zeros((H_high, W_high, 5), dtype=np.float32)
    
    if mask_high is not None:
        if mask_high.shape[:2] != (H_high, W_high):
            raise ValueError("أبعاد القناع لا تتطابق مع أبعاد الصورة المرفوعة.")
        
        bboxes = extract_bounding_boxes(mask_high, min_area=10)
        
        for (x, y, w, h) in bboxes:
            x_end = x + w
            y_end = y + h
            
            x_low = x // 2
            y_low = y // 2
            x_end_low = (x_end + 1) // 2
            y_end_low = (y_end + 1) // 2
            
            w_low = x_end_low - x_low
            h_low = y_end_low - y_low
            
            if w_low <= 0 or h_low <= 0:
                continue
            if x_end_low > W_low: x_end_low = W_low
            if y_end_low > H_low: y_end_low = H_low
            if x_low >= W_low or y_low >= H_low:
                continue
            
            patch_low = nnf_low[y_low:y_end_low, x_low:x_end_low, :]
            
            # ------------------------------------------------------------
            # التحسين رقم 1: الحصول على عرض (View) مباشر من مصفوفة النتيجة
            # واستخدامه كـ dst لكتابة المخرجات مباشرة دون تخصيص جديد.
            # ------------------------------------------------------------
            target_view = nnf_upsampled[y:y+h, x:x+w, :]
            cv2.resize(patch_low, (w, h), dst=target_view, interpolation=cv2.INTER_LINEAR)
            
            # تطبيق التحويل المكاني في نفس المكان (In-place) على target_view
            target_view[:, :, 0] *= scale_x
            target_view[:, :, 1] *= scale_y
    
    else:
        # الوضع الاحتياطي (بدون قناع): استخدام dst لكتابة النتيجة مباشرة
        for i in range(5):
            cv2.resize(nnf_low[:, :, i], (W_high, H_high), 
                       dst=nnf_upsampled[:, :, i], interpolation=cv2.INTER_LINEAR)
        nnf_upsampled[:, :, 0] *= scale_x
        nnf_upsampled[:, :, 1] *= scale_y
    
    return np.ascontiguousarray(nnf_upsampled)