"""
pipeline_core.py - نواة معالجة التبييض مع دعم ROI
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any

# استيراد الوحدات السابقة (سنفترض وجودها)
from mask_generator import generate_hybrid_mask
from engine import patchmatch_5d_hierarchical


class MangaProcessorPipeline:
    """
    معالج الصور مع دعم معالجة مناطق محددة (ROI).
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.verbose = self.config.get('verbose', False)
    
    def process_roi(
        self,
        image: np.ndarray,
        roi_mask: np.ndarray,
        roi_bbox: Tuple[int, int, int, int],
        mask_full: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        معالجة منطقة محددة فقط.
        
        المعاملات:
            image: الصورة الكاملة (float32 [0,1])
            roi_mask: الماسك الخاص بالمنطقة (بنفس أبعاد منطقة الـ ROI)
            roi_bbox: (x_min, y_min, x_max, y_max) موقع المنطقة في الصورة الكاملة
            mask_full: (اختياري) الماسك الكامل للصورة (للمعالجة الهرمية إذا لزم الأمر)
        
        المخرجات:
            (image_updated, nnf_result)
        """
        x_min, y_min, x_max, y_max = roi_bbox
        h = y_max - y_min
        w = x_max - x_min
        
        # استخراج المنطقة
        roi_image = image[y_min:y_max, x_min:x_max].copy()
        
        # إذا لم يكن الماسك الكامل موجوداً، نستخدم roi_mask كقناع كامل للمنطقة
        if mask_full is None:
            mask_full = np.zeros_like(image[:, :, 0], dtype=np.uint8)
            mask_full[y_min:y_max, x_min:x_max] = roi_mask
        
        # توليد قناع محسن باستخدام المولد الهجين (اختياري، يمكن تعطيله للتوفير)
        if self.config.get('use_hybrid_mask', True):
            # نمرر الـ ROI فقط لتوفير الوقت
            refined_mask = generate_hybrid_mask(
                roi_image,
                verbose=self.verbose,
                **self.config.get('mask_params', {})
            )
            # دمج القناع المحسن مع الماسك الأصلي
            combined_mask = np.logical_or(roi_mask, refined_mask).astype(np.uint8)
        else:
            combined_mask = roi_mask
        
        # تشغيل محرك PatchMatch على الـ ROI (مع تمرير الصورة الكاملة للمستوى الخشن؟)
        # الأفضل: نمرر الـ ROI للهرم، لكن نستفيد من الخلفية الكاملة في المستويات الخشنة؟
        # سنقوم بتمرير الصورة الكاملة للمحرك مع قناع محدود لتوفير سياق، لكن هذا يستهلك وقتاً.
        # الحل: نستخدم نسخة مصغرة من الصورة الكاملة للسياق، ونطبق المحرك على الـ ROI.
        # لكن سنبسط: نمرر الـ ROI والماسك الخاص به إلى المحرك الهرمي.
        # لاحظ أن المحرك الهرمي سيعمل على تحجيم الـ ROI، لكن قد يفتقر إلى السياق البعيد.
        # لتحسين النتيجة، يمكننا توسيع الـ ROI بهامش أكبر.
        
        # تطبيق PatchMatch الهرمي على الـ ROI
        result_roi, nnf_roi = patchmatch_5d_hierarchical(
            roi_image,
            combined_mask,
            levels=self.config.get('levels', 4),
            iters_per_level=self.config.get('iters_per_level', None),
            patch_radius=self.config.get('patch_radius', 2),
            early_stop_threshold=self.config.get('early_stop_threshold', 1e-4),
            verbose=self.verbose
        )
        
        # دمج النتيجة في الصورة الكاملة
        image_updated = image.copy()
        image_updated[y_min:y_max, x_min:x_max] = result_roi
        
        # إرجاع الصورة المحدثة وقناع المنطقة (للتحديث)
        return image_updated, combined_mask
    
    def process_full(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        معالجة الصورة كاملة (الوضع التقليدي).
        """
        return patchmatch_5d_hierarchical(
            image,
            mask,
            levels=self.config.get('levels', 4),
            iters_per_level=self.config.get('iters_per_level', None),
            patch_radius=self.config.get('patch_radius', 2),
            early_stop_threshold=self.config.get('early_stop_threshold', 1e-4),
            verbose=self.verbose
        )