"""
interactive_inpainting.py - تطبيق تفاعلي لإصلاح الصور باستخدام Streamlit و Canvas
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time

from streamlit_drawable_canvas import st_canvas

from roi_utils import extract_roi, merge_roi, compute_bbox_from_strokes
from pipeline_core import MangaProcessorPipeline

# إعدادات الصفحة
st.set_page_config(
    page_title="🖌️ مصحح المانجا التفاعلي",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# 1. تهيئة حالة الجلسة (Session State)
# ============================================================
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "mask" not in st.session_state:
    st.session_state.mask = None
if "strokes" not in st.session_state:
    st.session_state.strokes = []  # قائمة بضربات الفرشاة (كل ضربة قائمة نقاط)
if "undo_stack" not in st.session_state:
    st.session_state.undo_stack = []  # لتخزين الصور السابقة
if "redo_stack" not in st.session_state:
    st.session_state.redo_stack = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "pipeline" not in st.session_state:
    st.session_state.pipeline = MangaProcessorPipeline({
        'verbose': False,
        'levels': 4,
        'use_hybrid_mask': True,
        'mask_params': {
            'mser_delta': 5,
            'cone_angle_deg': 35.0,
        }
    })

# ============================================================
# 2. دوال مساعدة
# ============================================================
def load_image(uploaded_file):
    """تحميل الصورة وتحويلها إلى float32 [0,1] مع الاحتفاظ بنسخة للعرض"""
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        return img_float
    return None


def apply_inpainting(image, mask, strokes):
    """
    تطبيق عملية التبييض على المنطقة المحددة بواسطة الضربات.
    """
    if image is None or mask is None or not strokes:
        return image, mask
    
    # حساب Bounding Box من الضربات
    bbox = compute_bbox_from_strokes(strokes, image.shape[:2], padding=15)
    if bbox is None:
        return image, mask
    
    x_min, y_min, x_max, y_max = bbox
    # استخراج ROI
    roi_image = image[y_min:y_max, x_min:x_max].copy()
    roi_mask = mask[y_min:y_max, x_min:x_max].copy()
    
    # معالجة المنطقة باستخدام pipeline
    st.session_state.processing = True
    start = time.perf_counter()
    
    # استدعاء pipeline.process_roi
    updated_full, updated_mask = st.session_state.pipeline.process_roi(
        image, roi_mask, bbox, mask
    )
    
    elapsed = time.perf_counter() - start
    st.session_state.processing = False
    st.toast(f"✅ تم الإصلاح في {elapsed:.2f} ثانية", icon="✨")
    
    return updated_full, updated_mask


def add_to_undo():
    """إضافة الحالة الحالية إلى مكدس التراجع"""
    if st.session_state.current_image is not None:
        st.session_state.undo_stack.append({
            'image': st.session_state.current_image.copy(),
            'mask': st.session_state.mask.copy() if st.session_state.mask is not None else None
        })
        # تنظيف مكدس الإعادة عند القيام بعمل جديد
        st.session_state.redo_stack.clear()


def undo():
    """التراجع عن آخر عملية"""
    if st.session_state.undo_stack:
        state = st.session_state.undo_stack.pop()
        st.session_state.redo_stack.append({
            'image': st.session_state.current_image.copy(),
            'mask': st.session_state.mask.copy() if st.session_state.mask is not None else None
        })
        st.session_state.current_image = state['image']
        st.session_state.mask = state['mask']
        st.session_state.strokes = []  # مسح الضربات لأنها أصبحت غير ذات صلة
        st.rerun()


def redo():
    """إعادة العملية المتراجعة"""
    if st.session_state.redo_stack:
        state = st.session_state.redo_stack.pop()
        st.session_state.undo_stack.append({
            'image': st.session_state.current_image.copy(),
            'mask': st.session_state.mask.copy() if st.session_state.mask is not None else None
        })
        st.session_state.current_image = state['image']
        st.session_state.mask = state['mask']
        st.session_state.strokes = []
        st.rerun()


def reset_all():
    """إعادة تعيين كل شيء إلى الحالة الأولية"""
    if st.session_state.original_image is not None:
        st.session_state.current_image = st.session_state.original_image.copy()
        st.session_state.mask = np.zeros(st.session_state.original_image.shape[:2], dtype=np.uint8)
        st.session_state.strokes = []
        st.session_state.undo_stack = []
        st.session_state.redo_stack = []
        st.rerun()


# ============================================================
# 3. واجهة المستخدم
# ============================================================
st.title("🖌️ مصحح المانجا التفاعلي")
st.markdown("ارسم على المناطق التي تريد إصلاحها، ثم اضغط **إصلاح**.")

# عمودان: التحكم والعرض
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("📂 تحميل")
    uploaded_file = st.file_uploader("اختر صورة", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        img_float = load_image(uploaded_file)
        if st.session_state.original_image is None:
            st.session_state.original_image = img_float
            st.session_state.current_image = img_float.copy()
            st.session_state.mask = np.zeros(img_float.shape[:2], dtype=np.uint8)
            st.session_state.strokes = []
            st.session_state.undo_stack = []
            st.session_state.redo_stack = []
            st.rerun()
    
    if st.session_state.current_image is not None:
        st.subheader("🎨 أدوات")
        # ألوان الفرشاة
        brush_color = st.color_picker("لون الفرشاة", "#FF0000")
        brush_size = st.slider("حجم الفرشاة", 5, 50, 20)
        
        # أزرار التحكم
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button("↩️ تراجع", disabled=len(st.session_state.undo_stack)==0):
                undo()
        with col_btn2:
            if st.button("↪️ إعادة", disabled=len(st.session_state.redo_stack)==0):
                redo()
        with col_btn3:
            if st.button("🔄 إعادة تعيين"):
                reset_all()
        
        if st.button("🧹 مسح الفرشاة"):
            st.session_state.strokes = []
            st.rerun()
        
        if st.button("✨ إصلاح", disabled=st.session_state.processing or len(st.session_state.strokes)==0):
            if st.session_state.current_image is not None:
                add_to_undo()
                new_img, new_mask = apply_inpainting(
                    st.session_state.current_image,
                    st.session_state.mask,
                    st.session_state.strokes
                )
                st.session_state.current_image = new_img
                st.session_state.mask = new_mask
                st.session_state.strokes = []  # مسح الضربات بعد الإصلاح
                st.rerun()


with col2:
    if st.session_state.current_image is not None:
        # تحويل الصورة إلى uint8 للعرض
        img_display = (np.clip(st.session_state.current_image, 0, 1) * 255).astype(np.uint8)
        
        # تحويل الماسك إلى صورة ملونة للعرض كطبقة شفافة
        if st.session_state.mask is not None:
            mask_overlay = np.zeros_like(img_display)
            mask_overlay[:, :, 0] = (st.session_state.mask * 255).astype(np.uint8)  # قناة حمراء
            # مزج الماسك مع الصورة
            alpha = 0.3
            display_with_mask = cv2.addWeighted(img_display, 1 - alpha, mask_overlay, alpha, 0)
        else:
            display_with_mask = img_display
        
        # استخدام Canvas للرسم
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",  # لون التعبئة عند الرسم
            stroke_width=brush_size,
            stroke_color=brush_color,
            background_image=Image.fromarray(display_with_mask),
            update_streamlit=True,
            height=600,
            width=800,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        # معالجة أحداث الرسم
        if canvas_result is not None and canvas_result.json_data is not None:
            # استخراج الضربات الجديدة من canvas
            objects = canvas_result.json_data["objects"]
            new_strokes = []
            for obj in objects:
                if obj["type"] == "path":
                    # استخراج نقاط المسار
                    path_points = obj["path"]
                    stroke = []
                    for point in path_points:
                        if len(point) >= 3:
                            x = int(point[1])
                            y = int(point[2])
                            # التحقق من الحدود
                            if 0 <= x < st.session_state.current_image.shape[1] and 0 <= y < st.session_state.current_image.shape[0]:
                                stroke.append((x, y))
                    if stroke:
                        new_strokes.append(stroke)
            
            # تحديث حالة الضربات (نضيف الضربات الجديدة فقط)
            if new_strokes:
                # إضافة الضربات الجديدة إلى قائمة الضربات العامة
                st.session_state.strokes.extend(new_strokes)
                
                # تحديث الماسك (تسجيل المناطق المرسومة)
                if st.session_state.mask is not None:
                    for stroke in new_strokes:
                        for x, y in stroke:
                            # نرسم دائرة حول النقطة لتوسيع الماسك
                            cv2.circle(st.session_state.mask, (x, y), brush_size//2, 1, -1)
                
                # إعادة تشغيل التطبيق لتحديث العرض
                st.rerun()
    else:
        st.info("📤 قم بتحميل صورة للبدء.")


# ============================================================
# 4. شريط الحالة
# ============================================================
if st.session_state.processing:
    st.warning("⏳ جاري المعالجة...")
else:
    if st.session_state.current_image is not None:
        st.success(f"✅ الصورة جاهزة. عدد الضربات: {len(st.session_state.strokes)} | التراجع: {len(st.session_state.undo_stack)}")


# ============================================================
# 5. حفظ الصورة (في الأسفل)
# ============================================================
if st.session_state.current_image is not None:
    st.divider()
    col_save1, col_save2 = st.columns(2)
    with col_save1:
        # حفظ الصورة الناتجة
        img_save = (np.clip(st.session_state.current_image, 0, 1) * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_save)
        st.download_button(
            label="💾 تحميل الصورة المصححة",
            data=img_pil.tobytes(),
            file_name="inpainted_result.png",
            mime="image/png",
        )
    with col_save2:
        # حفظ الماسك
        if st.session_state.mask is not None:
            mask_save = (st.session_state.mask * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_save)
            st.download_button(
                label="📋 تحميل الماسك",
                data=mask_pil.tobytes(),
                file_name="mask.png",
                mime="image/png",
            )