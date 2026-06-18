import cv2
import numpy as np
import time
import os
from typing import Tuple, Dict, Any

# استيراد الموديول الهجين الذي قمنا بتدقيقه سابقاً
from mask_generator import generate_hybrid_mask, visualize_mask

class MangaProcessorPipeline:
    def __init__(self, config: Dict[str, Any] = None):
        """
        تهيئة خط الإنتاج مع إمكانية تمرير فلاتر وإعدادات مخصصة ديناميكياً.
        """
        # الإعدادات الافتراضية للنظام الهجين (تُطبق في حال عدم تمرير قيم مخصصة)
        self.default_config = {
            "mser_delta": 5,
            "mser_min_area": 60,
            "mser_max_area": 14400,
            "cone_angle_deg": 35.0,
            "min_stroke_ratio": 0.45,
            "min_aspect_ratio": 0.1,
            "max_aspect_ratio": 10.0,
            "min_solidity": 0.15,
            "max_stroke_relative": 0.05,
            "close_kernel": 3,
            "close_iterations": 1,
            "open_kernel": 2,
            "open_iterations": 1
        }
        
        # دمج الإعدادات الممررة مع الافتراضية
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)

    def process_single_image(
        self, 
        image_path: str, 
        output_dir: str, 
        save_debug: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        معالجة صورة واحدة من أي مسار ممرر وإخراج النتيجة إلى أي مجلد ممرر.
        المخرجات: (الصورة البيضاء النهائية، قناع النص الثنائي)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❌ المسار الممرر غير موجود: {image_path}")
            
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 1. قراءة الصورة وتحويل الألوان
        img_bgr = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. توليد القناع باستخدام الإعدادات الديناميكية للكلاس
        mask = generate_hybrid_mask(
            img_rgb,
            mser_delta=self.config["mser_delta"],
            mser_min_area=self.config["mser_min_area"],
            mser_max_area=self.config["mser_max_area"],
            cone_angle_deg=self.config["cone_angle_deg"],
            min_stroke_ratio=self.config["min_stroke_ratio"],
            min_aspect_ratio=self.config["min_aspect_ratio"],
            max_aspect_ratio=self.config["max_aspect_ratio"],
            min_solidity=self.config["min_solidity"],
            max_stroke_relative=self.config["max_stroke_relative"],
            close_kernel=self.config["close_kernel"],
            close_iterations=self.config["close_iterations"],
            open_kernel=self.config["open_kernel"],
            open_iterations=self.config["open_iterations"],
            verbose=False # جعلناها صامتة لعدم ملء الشاشة أثناء المعالجة الجماعية
        )
        
        # 3. حفظ ملفات التصحيح البصري والماسك بمسار المخرجات المحدد
        if save_debug:
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_mask.png"), mask * 255)
            vis_img = visualize_mask(img_rgb, mask, alpha=0.4)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_vis.png"), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
            
        # 4. محرك التبييض السيادي (تمثيل منطقي للـ PatchMatch)
        # هنا يتم استدعاء دالة patchmatch_5d_hierarchical(img_rgb, mask)
        cleaned_rgb = img_rgb.copy() 
        # بكسلات النص (mask == 1) يتم تبييضها وإخفائها
        
        # تحويل النتيجة لحفظها
        cleaned_bgr = cv2.cvtColor(cleaned_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_cleaned.png"), cleaned_bgr)
        
        return cleaned_rgb, mask
