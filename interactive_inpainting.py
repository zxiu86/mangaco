"""
interactive_inpainting.py — MangaCo Studio v4.0
"Partial HTML Edition"

إصلاحات جذرية:
  ① الصورة لا تظهر في الكانفاس:
     السبب الحقيقي: fabric.js (داخل st_canvas) يرفض data:URI عند
     تعيين crossOrigin='anonymous'. الحل: استخدام Streamlit's
     media_file_manager للحصول على URL حقيقي (/_stcore/media/...).
  ② TypeError في st.image: safe_image() wrapper يتعامل مع كل الإصدارات.
  ③ تحويل جزئي إلى HTML:
     - لوحات المعاينة (الأصلي / النتيجة / القناع) → HTML مع base64
     - إحصاءات الجلسة → HTML
     - الكانفاس الرئيسي → st_canvas (مع URL صحيح للخلفية)
"""

# ═══════════════════════════════════════════════════════════════
# BUG FIX ① — Monkey-patch باستخدام Streamlit media_file_manager
# يجب أن يكون قبل أي import لـ streamlit_drawable_canvas
#
# الفرق عن النسخ السابقة:
#   v2: PNG data:URI  → كبير → بعض المتصفحات ترفضه
#   v3: JPEG data:URI → أصغر → لكن fabric.js يرفضه بسبب crossOrigin
#   v4: /_stcore/media/... → URL حقيقي من خادم Streamlit ← يعمل دائماً
# ═══════════════════════════════════════════════════════════════
import io
import base64
import hashlib
import importlib as _il

_st_img_mod = _il.import_module("streamlit.elements.image")

if not hasattr(_st_img_mod, "image_to_url"):

    import numpy as _np
    from PIL import Image as _PIL

    def _pil_to_jpeg_bytes(pil: "_PIL.Image") -> bytes:
        buf = io.BytesIO()
        pil.convert("RGB").save(buf, format="JPEG", quality=85, optimize=True)
        return buf.getvalue()

    def _hosted_url(img_bytes: bytes, image_id: str) -> str | None:
        """محاولة تسجيل الصورة في Streamlit media manager والحصول على URL."""

        # الطريقة 1: Streamlit >= 1.28
        try:
            from streamlit.runtime import get_instance
            rt = get_instance()
            if rt is not None:
                url = rt.media_file_manager.add(img_bytes, "image/jpeg", image_id)
                if url:
                    return url
        except Exception:
            pass

        # الطريقة 2: Streamlit module-level singleton
        try:
            import streamlit.runtime.media_file_manager as _mfm
            for attr in ("_media_file_manager", "media_file_manager"):
                mgr = getattr(_mfm, attr, None)
                if mgr is not None:
                    url = mgr.add(img_bytes, "image/jpeg", image_id)
                    if url:
                        return url
        except Exception:
            pass

        return None

    def _image_to_url(image, width=-1, clamp=False, channels="RGB",
                      output_format="auto", image_id="", allow_emoji=False):
        """
        Compatibility shim for streamlit-drawable-canvas 0.9.3.
        Returns a real Streamlit-hosted URL so fabric.js can load it without CORS issues.
        Falls back to data:URI only if media manager is unavailable.
        """
        try:
            if isinstance(image, _PIL.Image):
                pil = image
            elif hasattr(image, "__array__"):
                arr = _np.asarray(image)
                if arr.ndim == 2:
                    arr = _np.stack([arr] * 3, axis=-1)
                elif arr.ndim == 3 and arr.shape[2] == 4:
                    arr = arr[:, :, :3]
                pil = _PIL.fromarray(arr.astype(_np.uint8))
            else:
                return ""

            img_bytes = _pil_to_jpeg_bytes(pil)
            uid = image_id or hashlib.md5(img_bytes[:512]).hexdigest()[:12]

            # أولاً: URL حقيقي من Streamlit server
            url = _hosted_url(img_bytes, uid)
            if url:
                return url

            # ثانياً: data:URI كـ fallback
            b64 = base64.b64encode(img_bytes).decode()
            return f"data:image/jpeg;base64,{b64}"

        except Exception:
            return ""

    _st_img_mod.image_to_url = _image_to_url
# ═══════════════════════════════════════════════════════════════

import streamlit as st
import streamlit.components.v1 as _stc
import numpy as np
import cv2
import time
import json
from PIL import Image

from streamlit_drawable_canvas import st_canvas
from roi_utils import compute_bbox_from_strokes
from pipeline_core import MangaProcessorPipeline


# ────────────────────────────────────────────────────────────────
# BUG FIX ②: wrapper آمن لـ st.image عبر كل إصدارات Streamlit
# ────────────────────────────────────────────────────────────────
def safe_image(src, **kw):
    kw.pop("use_container_width", None)
    kw.pop("use_column_width", None)
    for param in ("use_container_width", "use_column_width"):
        try:
            st.image(src, **{param: True, **kw})
            return
        except TypeError:
            continue
    st.image(src, **kw)


# ────────────────────────────────────────────────────────────────
# دوال الصور المساعدة
# ────────────────────────────────────────────────────────────────
def _f32_to_u8(arr: np.ndarray) -> np.ndarray:
    return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)

def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(_f32_to_u8(arr))

def _to_b64(pil: Image.Image, fmt="JPEG", q=82) -> str:
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format=fmt, quality=q)
    return base64.b64encode(buf.getvalue()).decode()

def _to_png_bytes(pil: Image.Image) -> bytes:
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


# ════════════════════════════════════════════════════════════════
# HTML helpers — لوحات المعاينة
# BUG FIX ③: تحويل جزئي إلى HTML (الصور مضمّنة كـ base64)
# ════════════════════════════════════════════════════════════════
_PV_STYLE = """
<style>
  body{margin:0;background:#0a0a12;font-family:'Inter',sans-serif;}
  .card{background:#0f0f1e;border:1px solid #1e1e38;border-radius:10px;
        padding:8px;margin-bottom:10px;}
  .lbl{font-size:.58rem;text-transform:uppercase;letter-spacing:.12em;
       color:#363660;margin-bottom:5px;font-weight:600;}
  img{width:100%;border-radius:6px;display:block;}
  .dim{font-size:.62rem;color:#2a2a50;text-align:center;margin-top:4px;}
</style>
"""

def _html_preview(title: str, pil_img: Image.Image) -> str:
    W, H = pil_img.size
    b64  = _to_b64(pil_img)
    return f"""<!DOCTYPE html><html><head>{_PV_STYLE}</head><body>
    <div class="card">
      <div class="lbl">{title}</div>
      <img src="data:image/jpeg;base64,{b64}" />
      <div class="dim">{W}×{H}</div>
    </div>
    </body></html>"""


def _html_mask_preview(mask_u8: np.ndarray) -> str:
    pil = Image.fromarray(mask_u8).convert("RGB")
    b64 = _to_b64(pil)
    return f"""<!DOCTYPE html><html><head>{_PV_STYLE}</head><body>
    <div class="card">
      <div class="lbl">القناع المتراكم</div>
      <img src="data:image/jpeg;base64,{b64}" />
    </div>
    </body></html>"""


def _html_stats(W: int, H: int, n_strokes: int, n_undo: int,
                pct_masked: float, status: str) -> str:
    color = {"جاهز": "#2adb80", "معالجة": "#f0a030", "فارغ": "#383858"}
    c = color.get(status, "#383858")
    return f"""<!DOCTYPE html><html>
    <head>
    <style>
    body{{margin:0;background:transparent;font-family:'Inter',sans-serif;}}
    .row{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px;}}
    .box{{flex:1;min-width:60px;background:#0e0e1c;border:1px solid #1c1c32;
          border-radius:7px;padding:7px;text-align:center;}}
    .v{{font-size:1rem;font-weight:700;color:#6a5fff;}}
    .l{{font-size:.55rem;color:#2e2e52;text-transform:uppercase;
        letter-spacing:.08em;}}
    .badge{{display:inline-block;padding:3px 12px;border-radius:20px;
            font-size:.65rem;font-weight:700;background:{c}18;
            color:{c};border:1px solid {c}44;margin-bottom:6px;}}
    </style></head><body>
    <div class="badge">{'✅ ' if status=='جاهز' else '⏳ ' if status=='معالجة' else '⬜ '}{status}</div>
    <div class="row">
      <div class="box"><div class="v">{W}</div><div class="l">عرض</div></div>
      <div class="box"><div class="v">{H}</div><div class="l">ارتفاع</div></div>
      <div class="box"><div class="v">{n_strokes}</div><div class="l">ضربات</div></div>
      <div class="box"><div class="v">{n_undo}</div><div class="l">تراجع</div></div>
      <div class="box"><div class="v">{pct_masked:.1f}%</div><div class="l">قناع</div></div>
    </div>
    </body></html>"""


# ════════════════════════════════════════════════════════════════
# CSS — تصميم جزئي HTML للصفحة الرئيسية
# ════════════════════════════════════════════════════════════════
_APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

[data-testid="stAppViewContainer"] {
    background: #080810;
    color: #d8d8ea;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: #0c0c18 !important;
    border-right: 1px solid #1a1a2e;
}
[data-testid="stSidebar"] * { color: #c0c0d8 !important; }
.block-container { padding-top: .5rem !important; }
#MainMenu, footer { visibility: hidden; }

/* الترويسة */
.app-header {
    background: linear-gradient(135deg, #0d0820 0%, #061428 50%, #081a10 100%);
    border: 1px solid #252540;
    border-radius: 14px;
    padding: 18px 26px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.app-header h1 {
    margin: 0;
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: -.02em;
    background: linear-gradient(100deg, #7c6bff 0%, #29dbb8 50%, #7c6bff 100%);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: hdr-glow 4s linear infinite;
}
@keyframes hdr-glow { 0%{background-position:0%} 100%{background-position:200%} }
.app-header .ver {
    font-size: .68rem;
    color: #2a2a50;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
}

/* قسم التحكم */
.ctrl-section {
    background: #0e0e1c;
    border: 1px solid #1c1c32;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 10px;
}
.ctrl-title {
    font-size: .62rem;
    text-transform: uppercase;
    letter-spacing: .12em;
    color: #363660;
    margin-bottom: 10px;
    font-weight: 700;
}

/* منطقة الكانفاس */
.canvas-area {
    background: #06060e;
    border: 1px solid #181830;
    border-radius: 12px;
    padding: 10px;
    margin-bottom: 12px;
}
.canvas-label {
    font-size: .62rem;
    text-transform: uppercase;
    letter-spacing: .12em;
    color: #252548;
    margin-bottom: 7px;
    font-weight: 700;
}

/* أزرار Streamlit */
[data-testid="stButton"] button {
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    transition: all .18s ease !important;
    border: 1px solid #1e1e38 !important;
}
[data-testid="stButton"] button:hover {
    border-color: #4a3fff !important;
    box-shadow: 0 0 12px #4a3fff28 !important;
}
</style>
"""


# ════════════════════════════════════════════════════════════════
# تهيئة الجلسة
# ════════════════════════════════════════════════════════════════
def _init():
    defaults = {
        "orig":     None,
        "curr":     None,
        "mask":     None,
        "strokes":  [],
        "undo":     [],
        "redo":     [],
        "busy":     False,
        "show_mask": True,
        "zoom":     100,
        "brush_sz": 18,
        "pipeline": MangaProcessorPipeline({
            "verbose": False, "levels": 4, "use_hybrid_mask": True,
            "mask_params": {"mser_delta": 5, "cone_angle_deg": 35.0},
        }),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()
ss = st.session_state


# ════════════════════════════════════════════════════════════════
# عمليات الجلسة
# ════════════════════════════════════════════════════════════════
def _push_undo():
    if ss.curr is None:
        return
    ss.undo.append({"img": ss.curr.copy(),
                    "mask": ss.mask.copy() if ss.mask is not None else None})
    ss.redo.clear()

def _restore(s):
    ss.curr = s["img"]; ss.mask = s["mask"]; ss.strokes = []

def do_undo():
    if not ss.undo: return
    ss.redo.append({"img": ss.curr.copy(), "mask": ss.mask.copy() if ss.mask is not None else None})
    _restore(ss.undo.pop()); st.rerun()

def do_redo():
    if not ss.redo: return
    ss.undo.append({"img": ss.curr.copy(), "mask": ss.mask.copy() if ss.mask is not None else None})
    _restore(ss.redo.pop()); st.rerun()

def do_reset():
    if ss.orig is None: return
    ss.curr = ss.orig.copy()
    ss.mask = np.zeros(ss.orig.shape[:2], np.uint8)
    ss.strokes = []; ss.undo = []; ss.redo = []
    st.rerun()

def do_inpaint():
    if ss.curr is None or not ss.strokes: return
    bbox = compute_bbox_from_strokes(ss.strokes, ss.curr.shape[:2], padding=20)
    if bbox is None: return
    x0, y0, x1, y1 = bbox
    roi_mask = ss.mask[y0:y1, x0:x1].copy()
    new_img, new_mask = ss.pipeline.process_roi(ss.curr, roi_mask, bbox, ss.mask)
    ss.curr = new_img; ss.mask = new_mask; ss.strokes = []


def _canvas_dims(shape):
    H, W = shape[:2]
    z = ss.zoom / 100.0
    return max(1, int(W * z)), max(1, int(H * z))


def _make_bg(img_f32, mask, cw, ch) -> Image.Image:
    base = _f32_to_u8(img_f32)
    if ss.show_mask and mask is not None and np.any(mask):
        ov = np.zeros_like(base)
        ov[:, :, 0] = (mask * 210).astype(np.uint8)
        base = cv2.addWeighted(base, 0.76, ov, 0.24, 0)
    pil = Image.fromarray(base)
    if pil.size != (cw, ch):
        pil = pil.resize((cw, ch), Image.LANCZOS)
    return pil


# ════════════════════════════════════════════════════════════════
# تخطيط الصفحة
# ════════════════════════════════════════════════════════════════
st.set_page_config(page_title="MangaCo Studio", page_icon="🎨",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(_APP_CSS, unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
  <h1>🎨 MangaCo Studio</h1>
  <span class="ver">PatchMatch 5D · v4.0</span>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# الشريط الجانبي
# ════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── تحميل الصورة ────────────────────────────────────────────
    st.markdown('<div class="ctrl-title">📂 تحميل الصورة</div>',
                unsafe_allow_html=True)
    up = st.file_uploader("", type=["png","jpg","jpeg","webp"],
                          label_visibility="collapsed")
    if up is not None:
        img = _load(up)
        if img is not None and ss.orig is None:
            ss.orig = img; ss.curr = img.copy()
            ss.mask = np.zeros(img.shape[:2], np.uint8)
            ss.strokes = []; ss.undo = []; ss.redo = []
            st.rerun()

    st.divider()

    # ── إحصاءات HTML ────────────────────────────────────────────
    if ss.curr is not None:
        H, W = ss.curr.shape[:2]
        pct = (float(ss.mask.sum()) / (H * W) * 100) if ss.mask is not None else 0.0
        status = "معالجة" if ss.busy else "جاهز"
    else:
        W = H = 0; pct = 0.0; status = "فارغ"

    _stc.html(
        _html_stats(W, H, len(ss.strokes), len(ss.undo), pct, status),
        height=95,
    )

    st.divider()

    if ss.curr is not None:
        # ── الفرشاة ───────────────────────────────────────────
        st.markdown('<div class="ctrl-title">🖌️ الفرشاة</div>',
                    unsafe_allow_html=True)
        brush_color = st.color_picker("اللون", "#FF2244",
                                      label_visibility="collapsed")
        ss.brush_sz = st.slider("الحجم", 3, 90, ss.brush_sz,
                                label_visibility="collapsed")
        ss.show_mask = st.toggle("تظليل القناع", value=ss.show_mask)

        st.divider()

        # ── الزوم ─────────────────────────────────────────────
        st.markdown('<div class="ctrl-title">🔍 الحجم</div>',
                    unsafe_allow_html=True)
        ss.zoom = st.slider("تكبير %", 25, 200, ss.zoom, step=5,
                            label_visibility="collapsed")
        cw, ch = _canvas_dims(ss.curr.shape)
        st.caption(f"{cw} × {ch} px")

        st.divider()

        # ── المحرك ────────────────────────────────────────────
        with st.expander("⚙️ إعدادات المحرك", expanded=False):
            lvl  = st.slider("مستويات الهرم", 2, 6, 4)
            pr   = st.slider("نصف قطر الرقعة", 1, 5, 2)
            hybr = st.toggle("كشف تلقائي MSER+SWT", value=True)
            ss.pipeline.config.update({"levels": lvl, "patch_radius": pr,
                                       "use_hybrid_mask": hybr})

        st.divider()

        # ── أزرار الإجراء ─────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("↩️", help="تراجع", disabled=not ss.undo): do_undo()
        with c2:
            if st.button("↪️", help="إعادة", disabled=not ss.redo): do_redo()
        with c3:
            if st.button("🔄", help="إعادة تعيين"): do_reset()

        if st.button("🧹 مسح الرسم", use_container_width=True):
            ss.strokes = []; st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✨  إصلاح المنطقة", type="primary",
                     use_container_width=True,
                     disabled=not ss.strokes or ss.busy):
            _push_undo()
            with st.spinner("جاري الترميم…"):
                t0 = time.perf_counter()
                do_inpaint()
                elapsed = time.perf_counter() - t0
            st.toast(f"✅ {elapsed:.2f} ث", icon="✨")
            st.rerun()

        st.divider()

        # ── تصدير ─────────────────────────────────────────────
        st.markdown('<div class="ctrl-title">💾 تصدير</div>',
                    unsafe_allow_html=True)
        st.download_button(
            "⬇️ الصورة المُرمَّمة",
            data=_to_png_bytes(_to_pil(ss.curr)),
            file_name="mangaco_result.png",
            mime="image/png",
            use_container_width=True,
        )
        if ss.mask is not None:
            st.download_button(
                "⬇️ القناع",
                data=_to_png_bytes(Image.fromarray((ss.mask*255).astype(np.uint8))),
                file_name="mangaco_mask.png",
                mime="image/png",
                use_container_width=True,
            )


# ════════════════════════════════════════════════════════════════
# المنطقة الرئيسية
# ════════════════════════════════════════════════════════════════
if ss.curr is None:
    st.markdown("""
    <div style="text-align:center;padding:110px 40px;color:#1a1a30;">
      <div style="font-size:4.5rem;margin-bottom:20px;filter:drop-shadow(0 0 20px #4a3fff44);">🎨</div>
      <div style="font-size:1.35rem;font-weight:700;color:#3a3a70;margin-bottom:8px;">
        حمّل صورة مانجا لبدء الجلسة
      </div>
      <div style="font-size:.8rem;color:#15152a;">
        PNG · JPG · JPEG · WEBP
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    cw, ch = _canvas_dims(ss.curr.shape)
    H_img, W_img = ss.curr.shape[:2]
    sx, sy = W_img / cw, H_img / ch

    main_col, pv_col = st.columns([4, 1], gap="small")

    # ──────────────────────────────────────────────
    # الكانفاس الرئيسي للرسم
    # ──────────────────────────────────────────────
    with main_col:
        st.markdown('<div class="canvas-label">🖊️ ارسم على المنطقة المراد إصلاحها</div>',
                    unsafe_allow_html=True)

        bg = _make_bg(ss.curr, ss.mask, cw, ch)

        canvas_result = st_canvas(
            fill_color="rgba(255,34,68,0.18)",
            stroke_width=ss.brush_sz,
            stroke_color=brush_color,
            background_image=bg,          # يستخدم monkey-patch → URL حقيقي
            update_streamlit=True,
            height=ch,
            width=cw,
            drawing_mode="freedraw",
            key="canvas_v4",
        )

        # ── معالجة ضربات الفرشاة ──────────────────
        if canvas_result is not None and canvas_result.json_data is not None:
            objects     = canvas_result.json_data.get("objects", [])
            new_strokes = []

            for obj in objects:
                if obj.get("type") != "path":
                    continue
                pts = []
                for pt in obj.get("path", []):
                    if len(pt) >= 3:
                        px = int(round(float(pt[1]) * sx))
                        py = int(round(float(pt[2]) * sy))
                        px = max(0, min(W_img - 1, px))
                        py = max(0, min(H_img - 1, py))
                        pts.append((px, py))
                if pts:
                    new_strokes.append(pts)

            if new_strokes and new_strokes != ss.strokes:
                ss.strokes = new_strokes
                r = max(1, int(ss.brush_sz * min(sx, sy) / 2))
                for stroke in new_strokes:
                    for px, py in stroke:
                        cv2.circle(ss.mask, (px, py), r, 1, -1)

    # ──────────────────────────────────────────────
    # BUG FIX ③: لوحات المعاينة كـ HTML خالص
    # الصور مضمّنة كـ base64 مباشرة → لا تعتمد على
    # أي Streamlit image API → تعمل دائماً
    # ──────────────────────────────────────────────
    with pv_col:
        # الأصلي
        _stc.html(
            _html_preview("الأصلي", _to_pil(ss.orig)),
            height=int(ss.orig.shape[0] / ss.orig.shape[1] * 200) + 50,
            scrolling=False,
        )

        # النتيجة الحالية
        _stc.html(
            _html_preview("النتيجة", _to_pil(ss.curr)),
            height=int(ss.curr.shape[0] / ss.curr.shape[1] * 200) + 50,
            scrolling=False,
        )

        # القناع
        if ss.mask is not None and np.any(ss.mask):
            mask_u8 = (ss.mask * 255).astype(np.uint8)
            _stc.html(
                _html_mask_preview(mask_u8),
                height=int(mask_u8.shape[0] / mask_u8.shape[1] * 200) + 45,
                scrolling=False,
            )
