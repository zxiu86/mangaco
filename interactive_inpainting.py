"""
interactive_inpainting.py — استوديو ترميم المانجا
النسخة النهائية v3.0

إصلاحات:
  ① monkey-patch مُحسَّن: PNG → JPEG (أصغر = canvas يظهر صحيح)
  ② safe_image(): يعمل مع كل إصدارات Streamlit (بدون TypeError)
  ③ حجم الكانفاس حر بالكامل + slider للتحكم (بدون تصغير إجباري)
  ④ download_button: io.BytesIO بدلاً من tobytes()
"""

# ═══════════════════════════════════════════════════════════════
# BUG FIX ①: Monkey-patch — يجب قبل أي import للـ canvas
#
# streamlit >= 1.28 حذفت image_to_url من streamlit.elements.image
# streamlit-drawable-canvas 0.9.3 تستدعيها في السطر 125
#
# السبب الجذري لعدم ظهور الصورة:
# - النسخة القديمة من الـ patch كانت تستخدم PNG → base64 كبير
# - بعض متصفحات/إصدارات Streamlit تفشل مع data-URI كبير للـ canvas
# الحل: تحويل إلى JPEG (أصغر 10x-20x من PNG)
# ═══════════════════════════════════════════════════════════════
import io
import base64
import importlib as _importlib

_st_img = _importlib.import_module("streamlit.elements.image")

if not hasattr(_st_img, "image_to_url"):
    import numpy as _np
    from PIL import Image as _PIL

    def _image_to_url(image, width=-1, clamp=False, channels="RGB",
                      output_format="auto", image_id="", allow_emoji=False):
        """
        Compatibility shim: converts PIL/ndarray to JPEG data-URI.
        JPEG is used (not PNG) to keep the URI small so browsers render it.
        """
        try:
            buf = io.BytesIO()
            if isinstance(image, _PIL.Image):
                pil = image.convert("RGB")
            elif hasattr(image, "__array__"):
                arr = _np.asarray(image)
                if arr.ndim == 2:
                    arr = _np.stack([arr] * 3, axis=-1)
                elif arr.shape[2] == 4:
                    arr = arr[:, :, :3]
                pil = _PIL.fromarray(arr.astype(_np.uint8))
            else:
                return ""
            pil.save(buf, format="JPEG", quality=82, optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"
        except Exception:
            return ""

    _st_img.image_to_url = _image_to_url
# ═══════════════════════════════════════════════════════════════

import streamlit as st
import numpy as np
import cv2
import time
from PIL import Image

from streamlit_drawable_canvas import st_canvas
from roi_utils import compute_bbox_from_strokes
from pipeline_core import MangaProcessorPipeline


# ═══════════════════════════════════════════════════════════════
# BUG FIX ②: wrapper آمن لـ st.image يعمل مع كل الإصدارات
# السبب: في بعض إصدارات Streamlit تغيّر اسم المعامل أو ترتيبه
# ═══════════════════════════════════════════════════════════════
def safe_image(src, **kwargs):
    """st.image() wrapper safe across all Streamlit versions."""
    # حذف أي kwargs غير معروفة من القاموس مسبقاً
    kwargs.pop("use_container_width", None)
    kwargs.pop("use_column_width", None)
    try:
        st.image(src, use_container_width=True, **kwargs)
    except TypeError:
        try:
            st.image(src, use_column_width=True, **kwargs)
        except TypeError:
            st.image(src, **kwargs)


# ────────────────────────────────────────────────────────────────
# CSS الاستوديو الداكن
# ────────────────────────────────────────────────────────────────
_CSS = """
<style>
[data-testid="stAppViewContainer"]          { background:#0d0d0f; color:#e0e0ea; }
[data-testid="stSidebar"]                   { background:#111119 !important;
                                              border-right:1px solid #23233a; }
[data-testid="stSidebar"] *                 { color:#c8c8dc !important; }
.block-container                            { padding-top:.6rem !important; }
#MainMenu, footer                           { visibility:hidden; }

/* ترويسة */
.hdr { background:linear-gradient(130deg,#160826,#081830,#081a10);
       border:1px solid #2a2a48; border-radius:12px;
       padding:16px 24px; margin-bottom:16px; }
.hdr h1 { margin:0; font-size:1.5rem; font-weight:700;
           background:linear-gradient(90deg,#8b7aff,#43dbb8,#8b7aff);
           background-size:200%;
           -webkit-background-clip:text; -webkit-text-fill-color:transparent;
           animation:sh 4s linear infinite; }
@keyframes sh { 0%{background-position:0%} 100%{background-position:200%} }
.hdr .sub { font-size:.75rem; color:#504870; margin-top:3px; }

/* بطاقة أداة */
.tc  { background:#141422; border:1px solid #222238;
       border-radius:9px; padding:12px 14px; margin-bottom:10px; }
.tct { font-size:.65rem; text-transform:uppercase; letter-spacing:.12em;
       color:#484870; font-weight:700; margin-bottom:8px; }

/* شارة */
.badge { display:inline-block; padding:2px 10px; border-radius:20px;
         font-size:.7rem; font-weight:700; }
.b-ok  { background:#082210; color:#3ad088; border:1px solid #154430; }
.b-run { background:#221008; color:#d0843a; border:1px solid #443015; }
.b-off { background:#141422; color:#484870; border:1px solid #222238; }

/* إحصاء */
.srow { display:flex; gap:8px; margin:8px 0; }
.sbox { flex:1; background:#0e0e1c; border:1px solid #1e1e32;
        border-radius:7px; padding:7px; text-align:center; }
.sbox .v { font-size:1.1rem; font-weight:700; color:#7060f0; }
.sbox .l { font-size:.6rem; color:#3a3a58; text-transform:uppercase;
           letter-spacing:.08em; }

/* معاينة */
.pvbox { background:#0c0c18; border:1px solid #1a1a30;
         border-radius:9px; padding:8px; margin-bottom:8px; }
.pvlbl { font-size:.6rem; color:#383858; text-transform:uppercase;
         letter-spacing:.1em; margin-bottom:5px; }

/* كانفاس */
.cvwrap { background:#080810; border:1px solid #1e1e34;
          border-radius:10px; padding:10px; }
.cvlbl  { font-size:.65rem; color:#383870; text-transform:uppercase;
           letter-spacing:.12em; margin-bottom:6px; }
</style>
"""


# ════════════════════════════════════════════════════════════════
# تهيئة الجلسة
# ════════════════════════════════════════════════════════════════
def _init():
    defs = {
        "orig":       None,
        "curr":       None,
        "mask":       None,
        "strokes":    [],
        "undo":       [],
        "redo":       [],
        "busy":       False,
        "show_mask":  True,
        "zoom":       100,          # BUG FIX ③: zoom بدلاً من حجم ثابت
        "brush_sz":   20,
        "pipeline":   MangaProcessorPipeline({
            "verbose": False, "levels": 4, "use_hybrid_mask": True,
            "mask_params": {"mser_delta": 5, "cone_angle_deg": 35.0},
        }),
    }
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()
ss = st.session_state          # اختصار


# ════════════════════════════════════════════════════════════════
# دوال مساعدة
# ════════════════════════════════════════════════════════════════
def _f2u(arr: np.ndarray) -> np.ndarray:
    """float32 [0,1] → uint8 [0,255]"""
    return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)


def _f2pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(_f2u(arr))


# BUG FIX ④: tobytes() كانت تعيد بيانات خام — نستخدم PNG حقيقي
def _to_png(pil: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _load(file) -> np.ndarray | None:
    if file is None:
        return None
    data = np.frombuffer(file.read(), np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _canvas_size(shape):
    """
    BUG FIX ③: بدون تصغير إجباري.
    الحجم = حجم الصورة × zoom%
    """
    H, W = shape[:2]
    z = ss.zoom / 100.0
    return max(1, int(W * z)), max(1, int(H * z))


def _push_undo():
    if ss.curr is None:
        return
    ss.undo.append({"img": ss.curr.copy(),
                    "mask": ss.mask.copy() if ss.mask is not None else None})
    ss.redo.clear()


def _restore(state):
    ss.curr    = state["img"]
    ss.mask    = state["mask"]
    ss.strokes = []


def do_undo():
    if not ss.undo:
        return
    ss.redo.append({"img": ss.curr.copy(),
                    "mask": ss.mask.copy() if ss.mask is not None else None})
    _restore(ss.undo.pop())
    st.rerun()


def do_redo():
    if not ss.redo:
        return
    ss.undo.append({"img": ss.curr.copy(),
                    "mask": ss.mask.copy() if ss.mask is not None else None})
    _restore(ss.redo.pop())
    st.rerun()


def do_reset():
    if ss.orig is None:
        return
    ss.curr    = ss.orig.copy()
    ss.mask    = np.zeros(ss.orig.shape[:2], np.uint8)
    ss.strokes = []
    ss.undo    = []
    ss.redo    = []
    st.rerun()


def do_inpaint():
    if ss.curr is None or not ss.strokes:
        return
    bbox = compute_bbox_from_strokes(ss.strokes, ss.curr.shape[:2], padding=20)
    if bbox is None:
        return
    x0, y0, x1, y1 = bbox
    roi_mask = ss.mask[y0:y1, x0:x1].copy()
    new_img, new_mask = ss.pipeline.process_roi(ss.curr, roi_mask, bbox, ss.mask)
    ss.curr    = new_img
    ss.mask    = new_mask
    ss.strokes = []


def _make_bg(img_f32: np.ndarray, mask: np.ndarray | None,
             cw: int, ch: int) -> Image.Image:
    """اصنع صورة الخلفية للكانفاس مع تظليل القناع."""
    base = _f2u(img_f32)
    if ss.show_mask and mask is not None and np.any(mask):
        ov           = np.zeros_like(base)
        ov[:, :, 0] = (mask * 210).astype(np.uint8)
        base         = cv2.addWeighted(base, 0.76, ov, 0.24, 0)
    pil = Image.fromarray(base)
    # تغيير الحجم حسب zoom
    if pil.size != (cw, ch):
        pil = pil.resize((cw, ch), Image.LANCZOS)
    return pil


# ════════════════════════════════════════════════════════════════
# تخطيط الصفحة
# ════════════════════════════════════════════════════════════════
st.set_page_config(page_title="MangaCo Studio", page_icon="🎨",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(_CSS, unsafe_allow_html=True)

st.markdown("""
<div class="hdr">
  <h1>🎨 MangaCo Studio</h1>
  <div class="sub">استوديو ترميم المانجا — PatchMatch 5D Engine</div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# الشريط الجانبي
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📂 الصورة")
    up = st.file_uploader("", type=["png","jpg","jpeg","webp"],
                          label_visibility="collapsed")

    if up is not None:
        img = _load(up)
        if img is not None and ss.orig is None:
            ss.orig    = img
            ss.curr    = img.copy()
            ss.mask    = np.zeros(img.shape[:2], np.uint8)
            ss.strokes = []
            ss.undo    = []
            ss.redo    = []
            st.rerun()

    st.divider()

    # شارة الحالة
    if ss.curr is None:
        st.markdown('<span class="badge b-off">⬜ لا توجد صورة</span>',
                    unsafe_allow_html=True)
    elif ss.busy:
        st.markdown('<span class="badge b-run">⏳ معالجة…</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge b-ok">✅ جاهز</span>',
                    unsafe_allow_html=True)

    if ss.curr is not None:
        H, W = ss.curr.shape[:2]
        st.markdown(f"""
        <div class="srow">
          <div class="sbox"><div class="v">{W}</div><div class="l">عرض</div></div>
          <div class="sbox"><div class="v">{H}</div><div class="l">ارتفاع</div></div>
          <div class="sbox"><div class="v">{len(ss.strokes)}</div><div class="l">ضربات</div></div>
          <div class="sbox"><div class="v">{len(ss.undo)}</div><div class="l">تراجع</div></div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    if ss.curr is not None:
        # ── الفرشاة ─────────────────────────────────────────
        st.markdown("### 🖌️ الفرشاة")
        brush_color = st.color_picker("اللون", "#FF3355",
                                      label_visibility="collapsed")
        ss.brush_sz = st.slider("الحجم", 3, 80, ss.brush_sz,
                                label_visibility="collapsed",
                                help="حجم الفرشاة بالبكسل")
        ss.show_mask = st.toggle("تظليل القناع", value=ss.show_mask)

        st.divider()

        # BUG FIX ③: تكبير/تصغير حر بالكامل
        st.markdown("### 🔍 العرض")
        ss.zoom = st.slider("تكبير %", 25, 200, ss.zoom, step=5,
                            help="100% = الحجم الطبيعي للصورة")
        cw, ch = _canvas_size(ss.curr.shape)
        st.caption(f"حجم الكانفاس: {cw} × {ch} px")

        st.divider()

        # ── إعدادات المحرك ───────────────────────────────────
        with st.expander("⚙️ المحرك", expanded=False):
            lvl  = st.slider("مستويات الهرم", 2, 6, 4)
            pr   = st.slider("نصف قطر الرقعة", 1, 5, 2)
            hybr = st.toggle("كشف تلقائي MSER+SWT", value=True)
            ss.pipeline.config.update({
                "levels": lvl, "patch_radius": pr,
                "use_hybrid_mask": hybr,
            })

        st.divider()

        # ── أزرار الإجراء ────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("↩️", help="تراجع", disabled=not ss.undo):
                do_undo()
        with c2:
            if st.button("↪️", help="إعادة", disabled=not ss.redo):
                do_redo()
        with c3:
            if st.button("🔄", help="إعادة تعيين الكل"):
                do_reset()

        if st.button("🧹 مسح الرسم", use_container_width=True):
            ss.strokes = []
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        can_fix = bool(ss.strokes) and not ss.busy
        if st.button("✨  إصلاح المنطقة", type="primary",
                     use_container_width=True, disabled=not can_fix):
            _push_undo()
            with st.spinner("جاري الترميم…"):
                t0 = time.perf_counter()
                do_inpaint()
                elapsed = time.perf_counter() - t0
            st.toast(f"✅ اكتمل في {elapsed:.2f} ث", icon="✨")
            st.rerun()

        st.divider()

        # ── تصدير ────────────────────────────────────────────
        st.markdown("### 💾 تصدير")
        st.download_button(
            "⬇️ الصورة المُرمَّمة",
            data=_to_png(_f2pil(ss.curr)),
            file_name="mangaco_result.png",
            mime="image/png",
            use_container_width=True,
        )
        if ss.mask is not None:
            mask_pil = Image.fromarray((ss.mask * 255).astype(np.uint8))
            st.download_button(
                "⬇️ القناع",
                data=_to_png(mask_pil),
                file_name="mangaco_mask.png",
                mime="image/png",
                use_container_width=True,
            )


# ════════════════════════════════════════════════════════════════
# المنطقة الرئيسية
# ════════════════════════════════════════════════════════════════
if ss.curr is None:
    st.markdown("""
    <div style="text-align:center;padding:100px 40px;color:#303050;">
      <div style="font-size:5rem;margin-bottom:24px;">🎨</div>
      <div style="font-size:1.4rem;font-weight:700;color:#5050a0;">
        حمّل صورة مانجا من الشريط الجانبي
      </div>
      <div style="font-size:.85rem;color:#252540;margin-top:10px;">
        PNG · JPG · JPEG · WEBP
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    cw, ch = _canvas_size(ss.curr.shape)
    H_img, W_img = ss.curr.shape[:2]
    scale_x = W_img / cw
    scale_y = H_img / ch

    main_col, side_col = st.columns([4, 1], gap="medium")

    # ════════════════════════
    # الكانفاس الرئيسي
    # ════════════════════════
    with main_col:
        st.markdown(
            '<div class="cvlbl">🖊️ ارسم على المناطق المراد إصلاحها</div>',
            unsafe_allow_html=True,
        )

        bg_pil = _make_bg(ss.curr, ss.mask, cw, ch)

        canvas_result = st_canvas(
            fill_color="rgba(255,51,85,0.2)",
            stroke_width=ss.brush_sz,
            stroke_color=brush_color,
            background_image=bg_pil,
            update_streamlit=True,
            height=ch,
            width=cw,
            drawing_mode="freedraw",
            key="canvas_main",
        )

        # ── معالجة ضربات الفرشاة ──────────────────────────
        if canvas_result is not None and canvas_result.json_data is not None:
            objects     = canvas_result.json_data.get("objects", [])
            new_strokes = []

            for obj in objects:
                if obj.get("type") != "path":
                    continue
                pts = []
                for pt in obj.get("path", []):
                    if len(pt) >= 3:
                        px = int(round(float(pt[1]) * scale_x))
                        py = int(round(float(pt[2]) * scale_y))
                        px = max(0, min(W_img - 1, px))
                        py = max(0, min(H_img - 1, py))
                        pts.append((px, py))
                if pts:
                    new_strokes.append(pts)

            if new_strokes and new_strokes != ss.strokes:
                ss.strokes = new_strokes
                # رسم على القناع
                r = max(1, int(ss.brush_sz * min(scale_x, scale_y) / 2))
                for stroke in new_strokes:
                    for px, py in stroke:
                        cv2.circle(ss.mask, (px, py), r, 1, -1)

    # ════════════════════════
    # عمود المعاينة
    # ════════════════════════
    with side_col:
        # ── الأصلي ──────────────────────────────────────
        st.markdown('<div class="pvbox">', unsafe_allow_html=True)
        st.markdown('<div class="pvlbl">الأصلي</div>', unsafe_allow_html=True)
        if ss.orig is not None:
            safe_image(_f2pil(ss.orig))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── النتيجة الحالية ─────────────────────────────
        st.markdown('<div class="pvbox">', unsafe_allow_html=True)
        st.markdown('<div class="pvlbl">النتيجة</div>', unsafe_allow_html=True)
        safe_image(_f2pil(ss.curr))
        st.markdown('</div>', unsafe_allow_html=True)

        # ── القناع ──────────────────────────────────────
        if ss.mask is not None and np.any(ss.mask):
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="pvbox">', unsafe_allow_html=True)
            st.markdown('<div class="pvlbl">القناع</div>', unsafe_allow_html=True)
            mask_u8 = (ss.mask * 255).astype(np.uint8)
            safe_image(mask_u8)
            st.markdown('</div>', unsafe_allow_html=True)
