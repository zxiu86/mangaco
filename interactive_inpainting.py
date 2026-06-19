"""
interactive_inpainting.py — استوديو ترميم المانجا التفاعلي
النسخة المستقرة v2.0 — تصميم Studio Dark

إصلاحات:
  ① monkey-patch لـ streamlit.elements.image.image_to_url
     (حل AttributeError مع Streamlit >= 1.28)
  ② إصلاح download_button: tobytes() ← io.BytesIO + save()
  ③ إعادة تصميم كاملة بتجربة استوديو احترافية
"""

# ─────────────────────────────────────────────────────────────
# BUG FIX ①: Monkey-patch BEFORE importing streamlit_drawable_canvas
# streamlit >= 1.28 removes `image_to_url` from streamlit.elements.image
# which streamlit-drawable-canvas 0.9.3 needs internally.
# We restore it as a simple base64 data-URI converter.
# ─────────────────────────────────────────────────────────────
import io
import base64
import streamlit.elements.image as _st_img_module
from PIL import Image as _PILImage

if not hasattr(_st_img_module, "image_to_url"):
    def _image_to_url(image, width=-1, clamp=False, channels="RGB",
                      output_format="auto", image_id="", allow_emoji=False):
        buf = io.BytesIO()
        if isinstance(image, _PILImage.Image):
            fmt = "PNG"
            image.save(buf, format=fmt)
        elif hasattr(image, "tobytes"):                     # numpy array
            _PILImage.fromarray(image).save(buf, format="PNG")
        else:
            return ""
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    _st_img_module.image_to_url = _image_to_url
# ─────────────────────────────────────────────────────────────

import streamlit as st
import numpy as np
import cv2
import time
from PIL import Image

from streamlit_drawable_canvas import st_canvas
from roi_utils import compute_bbox_from_strokes
from pipeline_core import MangaProcessorPipeline


# ============================================================
# CSS — تصميم الاستوديو الداكن
# ============================================================
STUDIO_CSS = """
<style>
/* ── الخلفية العامة ──────────────────────────── */
[data-testid="stAppViewContainer"] {
    background: #0d0d0f;
    color: #e8e8ec;
}
[data-testid="stSidebar"] {
    background: #13131a !important;
    border-right: 1px solid #2a2a3a;
}
[data-testid="stSidebar"] * { color: #d0d0e0 !important; }

/* ── ترويسة الصفحة ───────────────────────────── */
.studio-header {
    background: linear-gradient(135deg, #1a0a2e 0%, #0d1b3e 50%, #0a2e1a 100%);
    border: 1px solid #3a3a5a;
    border-radius: 12px;
    padding: 18px 28px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.studio-header h1 {
    margin: 0;
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #7c6bff, #4ae0c4, #7c6bff);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite;
}
@keyframes shimmer { 0%{background-position:0%} 100%{background-position:200%} }
.studio-header .sub { font-size: 0.8rem; color: #6a6a8a; margin-top: 2px; }

/* ── بطاقات الأدوات ──────────────────────────── */
.tool-card {
    background: #1a1a2a;
    border: 1px solid #2a2a40;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 12px;
}
.tool-card-title {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6060a0;
    margin-bottom: 10px;
    font-weight: 600;
}

/* ── شارات الحالة ────────────────────────────── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
}
.badge-ready   { background: #0a2e1a; color: #4ae0a0; border: 1px solid #1a5a3a; }
.badge-busy    { background: #2e1a0a; color: #e0a04a; border: 1px solid #5a3a1a; }
.badge-empty   { background: #1a1a2a; color: #6060a0; border: 1px solid #2a2a40; }

/* ── شريط الإحصاء ────────────────────────────── */
.stat-row {
    display: flex;
    gap: 10px;
    margin-bottom: 14px;
    flex-wrap: wrap;
}
.stat-box {
    flex: 1;
    min-width: 80px;
    background: #111120;
    border: 1px solid #25253a;
    border-radius: 8px;
    padding: 8px 10px;
    text-align: center;
}
.stat-box .val { font-size: 1.2rem; font-weight: 700; color: #7c6bff; }
.stat-box .lbl { font-size: 0.65rem; color: #505070; text-transform: uppercase; letter-spacing: 0.08em; }

/* ── أزرار الإجراء ───────────────────────────── */
[data-testid="stButton"] button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

/* ── قسم الكانفاس ────────────────────────────── */
.canvas-wrapper {
    background: #0a0a12;
    border: 1px solid #2a2a40;
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 14px;
}
.canvas-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #5050a0;
    margin-bottom: 8px;
}

/* ── حاوية المعاينة ──────────────────────────── */
.preview-box {
    background: #0f0f1a;
    border: 1px solid #1e1e35;
    border-radius: 10px;
    padding: 10px;
    text-align: center;
}
.preview-box .preview-label {
    font-size: 0.65rem;
    color: #404060;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}

/* ── إخفاء العناصر الزائدة ───────────────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
.block-container { padding-top: 1rem !important; }
</style>
"""


# ============================================================
# تهيئة حالة الجلسة
# ============================================================
def _init_state():
    defaults = {
        "original_image":  None,
        "current_image":   None,
        "mask":            None,
        "strokes":         [],
        "undo_stack":      [],
        "redo_stack":      [],
        "processing":      False,
        "show_mask":       True,
        "brush_size":      20,
        "canvas_width":    800,
        "canvas_height":   600,
        "pipeline":        MangaProcessorPipeline({
            "verbose":        False,
            "levels":         4,
            "use_hybrid_mask": True,
            "mask_params": {
                "mser_delta":    5,
                "cone_angle_deg": 35.0,
            },
        }),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ============================================================
# دوال مساعدة
# ============================================================
def _to_uint8(img_f32: np.ndarray) -> np.ndarray:
    return (np.clip(img_f32, 0.0, 1.0) * 255).astype(np.uint8)


def _to_pil(img_f32: np.ndarray) -> Image.Image:
    return Image.fromarray(_to_uint8(img_f32))


# BUG FIX ②: الأصل كان img_pil.tobytes() → بيانات خام غير قابلة للفتح
# الصح: نحفظ بصيغة PNG داخل BytesIO
def _pil_to_png_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def _get_canvas_size(shape):
    H, W = shape[:2]
    ratio = min(800 / W, 560 / H, 1.0)
    return int(W * ratio), int(H * ratio)


def load_image(uploaded_file) -> np.ndarray | None:
    if uploaded_file is None:
        return None
    data = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _push_undo():
    if st.session_state.current_image is None:
        return
    st.session_state.undo_stack.append({
        "image": st.session_state.current_image.copy(),
        "mask":  st.session_state.mask.copy() if st.session_state.mask is not None else None,
    })
    st.session_state.redo_stack.clear()


def _restore(state: dict):
    st.session_state.current_image = state["image"]
    st.session_state.mask          = state["mask"]
    st.session_state.strokes       = []


def undo():
    if not st.session_state.undo_stack:
        return
    st.session_state.redo_stack.append({
        "image": st.session_state.current_image.copy(),
        "mask":  st.session_state.mask.copy() if st.session_state.mask is not None else None,
    })
    _restore(st.session_state.undo_stack.pop())
    st.rerun()


def redo():
    if not st.session_state.redo_stack:
        return
    st.session_state.undo_stack.append({
        "image": st.session_state.current_image.copy(),
        "mask":  st.session_state.mask.copy() if st.session_state.mask is not None else None,
    })
    _restore(st.session_state.redo_stack.pop())
    st.rerun()


def reset_all():
    if st.session_state.original_image is None:
        return
    st.session_state.current_image = st.session_state.original_image.copy()
    st.session_state.mask          = np.zeros(st.session_state.original_image.shape[:2], np.uint8)
    st.session_state.strokes       = []
    st.session_state.undo_stack    = []
    st.session_state.redo_stack    = []
    st.rerun()


def apply_inpainting(image, mask, strokes):
    if image is None or not strokes:
        return image, mask
    bbox = compute_bbox_from_strokes(strokes, image.shape[:2], padding=20)
    if bbox is None:
        return image, mask
    x_min, y_min, x_max, y_max = bbox
    roi_mask = mask[y_min:y_max, x_min:x_max].copy()
    updated, new_mask = st.session_state.pipeline.process_roi(image, roi_mask, bbox, mask)
    return updated, new_mask


def _make_display(img_f32, mask):
    """الصورة مع تظليل القناع الاختياري."""
    base = _to_uint8(img_f32)
    if st.session_state.show_mask and mask is not None and np.any(mask):
        overlay = np.zeros_like(base)
        overlay[:, :, 0] = (mask * 200).astype(np.uint8)
        base = cv2.addWeighted(base, 0.75, overlay, 0.25, 0)
    return Image.fromarray(base)


# ============================================================
# تخطيط الصفحة
# ============================================================
st.set_page_config(
    page_title="MangaCo Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(STUDIO_CSS, unsafe_allow_html=True)

# ── ترويسة ──────────────────────────────────────────────────
st.markdown("""
<div class="studio-header">
  <div>
    <h1>🎨 MangaCo Studio</h1>
    <div class="sub">استوديو ترميم المانجا التفاعلي — PatchMatch 5D Engine</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# الشريط الجانبي
# ============================================================
with st.sidebar:
    st.markdown("### 📂 تحميل الصورة")
    uploaded = st.file_uploader("", type=["png", "jpg", "jpeg", "webp"],
                                label_visibility="collapsed")

    if uploaded is not None:
        img_f = load_image(uploaded)
        if img_f is not None and st.session_state.original_image is None:
            st.session_state.original_image = img_f
            st.session_state.current_image  = img_f.copy()
            st.session_state.mask           = np.zeros(img_f.shape[:2], np.uint8)
            cw, ch = _get_canvas_size(img_f.shape)
            st.session_state.canvas_width   = cw
            st.session_state.canvas_height  = ch
            st.rerun()

    st.divider()

    # ── الحالة ──────────────────────────────────────────────
    if st.session_state.current_image is None:
        st.markdown('<span class="badge badge-empty">⬜ لا توجد صورة</span>',
                    unsafe_allow_html=True)
    elif st.session_state.processing:
        st.markdown('<span class="badge badge-busy">⏳ جاري المعالجة…</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-ready">✅ جاهز</span>',
                    unsafe_allow_html=True)

    if st.session_state.current_image is not None:
        H, W = st.session_state.current_image.shape[:2]
        n_strokes = len(st.session_state.strokes)
        n_undo    = len(st.session_state.undo_stack)
        st.markdown(f"""
        <div class="stat-row" style="margin-top:10px">
          <div class="stat-box"><div class="val">{W}×{H}</div><div class="lbl">دقة</div></div>
          <div class="stat-box"><div class="val">{n_strokes}</div><div class="lbl">ضربات</div></div>
          <div class="stat-box"><div class="val">{n_undo}</div><div class="lbl">تراجع</div></div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── أدوات الفرشاة ───────────────────────────────────────
    if st.session_state.current_image is not None:
        st.markdown("### 🖌️ الفرشاة")
        brush_color = st.color_picker("اللون", "#FF3355", label_visibility="collapsed")
        brush_size  = st.slider("الحجم", 3, 60, st.session_state.brush_size,
                                label_visibility="collapsed",
                                help="حجم الفرشاة")
        st.session_state.brush_size = brush_size
        st.session_state.show_mask  = st.toggle("إظهار القناع", value=st.session_state.show_mask)

        st.divider()

        # ── إعدادات المحرك ──────────────────────────────────
        with st.expander("⚙️ إعدادات المحرك", expanded=False):
            levels = st.slider("مستويات الهرم", 2, 6, 4)
            patch_r = st.slider("نصف قطر الرقعة", 1, 5, 2)
            use_hybrid = st.toggle("كشف تلقائي (MSER+SWT)", value=True)
            st.session_state.pipeline.config.update({
                "levels":         levels,
                "patch_radius":   patch_r,
                "use_hybrid_mask": use_hybrid,
            })

        st.divider()

        # ── أزرار الإجراء ────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("↩️", help="تراجع",
                         disabled=not st.session_state.undo_stack):
                undo()
        with c2:
            if st.button("↪️", help="إعادة",
                         disabled=not st.session_state.redo_stack):
                redo()
        with c3:
            if st.button("🔄", help="إعادة تعيين"):
                reset_all()

        if st.button("🧹 مسح الرسم", use_container_width=True):
            st.session_state.strokes = []
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        fix_disabled = (
            st.session_state.processing or
            not st.session_state.strokes
        )
        if st.button("✨  إصلاح المنطقة", type="primary",
                     use_container_width=True, disabled=fix_disabled):
            _push_undo()
            with st.spinner("جاري الترميم…"):
                t0 = time.perf_counter()
                new_img, new_mask = apply_inpainting(
                    st.session_state.current_image,
                    st.session_state.mask,
                    st.session_state.strokes,
                )
                elapsed = time.perf_counter() - t0
            st.session_state.current_image = new_img
            st.session_state.mask          = new_mask
            st.session_state.strokes       = []
            st.toast(f"✅ اكتمل في {elapsed:.2f} ث", icon="✨")
            st.rerun()

        st.divider()

        # ── تحميل ────────────────────────────────────────────
        st.markdown("### 💾 تصدير")
        result_pil = _to_pil(st.session_state.current_image)

        # BUG FIX ②: كان tobytes() — يعطي بيانات خام تفسد الملف
        st.download_button(
            "⬇️ تحميل الصورة المُرمَّمة",
            data=_pil_to_png_bytes(result_pil),
            file_name="mangaco_result.png",
            mime="image/png",
            use_container_width=True,
        )
        if st.session_state.mask is not None:
            mask_pil = Image.fromarray((st.session_state.mask * 255).astype(np.uint8))
            st.download_button(
                "⬇️ تحميل القناع",
                data=_pil_to_png_bytes(mask_pil),
                file_name="mangaco_mask.png",
                mime="image/png",
                use_container_width=True,
            )


# ============================================================
# المنطقة الرئيسية
# ============================================================
if st.session_state.current_image is None:
    # ── شاشة الترحيب ────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 80px 40px; color: #404060;">
      <div style="font-size:4rem; margin-bottom:20px;">🎨</div>
      <div style="font-size:1.3rem; font-weight:600; color:#6060a0; margin-bottom:10px;">
        ابدأ بتحميل صورة مانجا
      </div>
      <div style="font-size:0.85rem; color:#303050;">
        PNG · JPG · WEBP — الرسم على المنطقة المراد إصلاحها ثم اضغط ✨ إصلاح
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── عمودا العمل ─────────────────────────────────────────
    main_col, preview_col = st.columns([3, 1], gap="medium")

    with main_col:
        st.markdown('<div class="canvas-label">🖊️ منطقة الرسم — ارسم على النصوص المراد إزالتها</div>',
                    unsafe_allow_html=True)

        # تحضير صورة الخلفية
        display_pil = _make_display(
            st.session_state.current_image,
            st.session_state.mask,
        )
        # تكبير/تصغير للكانفاس
        if display_pil.size != (st.session_state.canvas_width,
                                 st.session_state.canvas_height):
            display_pil = display_pil.resize(
                (st.session_state.canvas_width, st.session_state.canvas_height),
                Image.LANCZOS,
            )

        canvas_result = st_canvas(
            fill_color=f"rgba(255,51,85,0.25)",
            stroke_width=st.session_state.brush_size,
            stroke_color=brush_color if st.session_state.current_image is not None else "#FF3355",
            background_image=display_pil,
            update_streamlit=True,
            height=st.session_state.canvas_height,
            width=st.session_state.canvas_width,
            drawing_mode="freedraw",
            key="main_canvas",
        )

        # معالجة ضربات الفرشاة
        if canvas_result is not None and canvas_result.json_data is not None:
            objects    = canvas_result.json_data.get("objects", [])
            new_strokes = []

            H_img, W_img = st.session_state.current_image.shape[:2]
            cw = st.session_state.canvas_width
            ch = st.session_state.canvas_height
            scale_x = W_img / cw
            scale_y = H_img / ch

            for obj in objects:
                if obj.get("type") != "path":
                    continue
                pts = []
                for pt in obj.get("path", []):
                    if len(pt) >= 3:
                        px = int(pt[1] * scale_x)
                        py = int(pt[2] * scale_y)
                        if 0 <= px < W_img and 0 <= py < H_img:
                            pts.append((px, py))
                if pts:
                    new_strokes.append(pts)

            if new_strokes != st.session_state.strokes[-len(new_strokes):] if new_strokes else False:
                pass

            # تحديث القناع
            if new_strokes:
                r = max(1, int(st.session_state.brush_size * min(scale_x, scale_y) / 2))
                for stroke in new_strokes:
                    for px, py in stroke:
                        cv2.circle(st.session_state.mask, (px, py), r, 1, -1)
                st.session_state.strokes = new_strokes

    with preview_col:
        # ── معاينة الأصلي ────────────────────────────────────
        st.markdown('<div class="preview-box">', unsafe_allow_html=True)
        st.markdown('<div class="preview-label">الأصلي</div>', unsafe_allow_html=True)
        if st.session_state.original_image is not None:
            st.image(_to_pil(st.session_state.original_image), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── معاينة النتيجة الحالية ────────────────────────────
        st.markdown('<div class="preview-box">', unsafe_allow_html=True)
        st.markdown('<div class="preview-label">النتيجة الحالية</div>', unsafe_allow_html=True)
        st.image(_to_pil(st.session_state.current_image), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── معاينة القناع المتراكم ────────────────────────────
        if st.session_state.mask is not None and np.any(st.session_state.mask):
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="preview-box">', unsafe_allow_html=True)
            st.markdown('<div class="preview-label">القناع المتراكم</div>', unsafe_allow_html=True)
            mask_vis = (st.session_state.mask * 255).astype(np.uint8)
            st.image(mask_vis, clamp=True, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
