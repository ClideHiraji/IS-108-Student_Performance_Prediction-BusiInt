from __future__ import annotations

import base64
import importlib.util
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image
from sklearn.metrics import confusion_matrix

import dataset_handling as data_io
import prediction as predictor
import train_models as model_training

TF_AVAILABLE = model_training.TF_AVAILABLE
TF_IMPORT_ERROR = model_training.TF_IMPORT_ERROR

PREPROCESSING_PATH = Path(__file__).with_name("pre-processing.py")
spec = importlib.util.spec_from_file_location("pre_processing", PREPROCESSING_PATH)
pre_processing = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = pre_processing
spec.loader.exec_module(pre_processing)

LOGO_PATH = Path(__file__).with_name("Logo.png")


def logo_data_uri(path: Path) -> str:
    """Return a cropped PNG data URI so transparent logo padding does not affect layout."""
    if not path.exists():
        return ""

    image = Image.open(path).convert("RGBA")
    alpha_bbox = image.getchannel("A").getbbox()
    if alpha_bbox:
        image = image.crop(alpha_bbox)

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"

st.set_page_config(
    page_title="Student Performance BI System",
    page_icon="SP",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── colour palette ──────────────────────────────────────────────────────────
DARK = {
    "bg":       "#141414",
    "surface":  "#1e1e1e",
    "surface_2":"#2a2520",
    "border":   "#3c3c3c",
    "accent":   "#c0622d",
    "muted":    "#a1887d",
    "text":     "#edeff3",
    "shadow":   "rgba(0,0,0,0.45)",
}
CHART = {
    "bg":      DARK["surface"],
    "panel":   DARK["surface_2"],
    "grid":    DARK["border"],
    "text":    DARK["text"],
    "muted":   DARK["muted"],
    "accent":  DARK["accent"],
    "accent_2":"#4a8fd4",
    "accent_3":DARK["muted"],
}

# ── chart sizes ─────────────────────────────────────────────────────────────
CHART_W,  CHART_H  = 5.5, 3.4   # general / dataset
GRADE_W,  GRADE_H  = 4.9, 2.85  # paired GradeClass charts
PREP_W,   PREP_H   = 4.0, 3.0   # 3-col preprocessing charts
EVAL_W,   EVAL_H   = 3.8, 3.2   # 4-col evaluation charts

# ════════════════════════════════════════════════════════════════════════════
def inject_css() -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
  --bg:        {DARK["bg"]};
  --surface:   {DARK["surface"]};
  --surface-2: {DARK["surface_2"]};
  --border:    {DARK["border"]};
  --input-border: #302d2a;
  --accent:    {DARK["accent"]};
  --muted:     {DARK["muted"]};
  --text:      {DARK["text"]};
  --shadow:    {DARK["shadow"]};
}}

/* ── global reset ── */
html, body, [class*="css"] {{
  font-family: Inter, system-ui, sans-serif;
  color: var(--text);
}}
.stApp {{
  background:
    radial-gradient(900px 420px at 18% -12%,
      color-mix(in srgb, var(--accent) 14%, transparent), transparent 64%),
    linear-gradient(180deg, var(--bg), var(--surface));
  color: var(--text);
}}
.main .block-container {{
  max-width: 1420px;
  padding-top: 0.2rem;
  padding-bottom: 2rem;
}}
[data-testid="stAppViewContainer"] .main .block-container,
[data-testid="stAppViewContainer"] section.main > div {{
  padding-top: 0.2rem !important;
}}
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewContainer"] [class*="block-container"],
[data-testid="stAppViewContainer"] [class*="stMainBlockContainer"],
.st-emotion-cache-zy6yx3 {{
  padding-top: 2rem !important;
  padding-right: 6rem !important;
  padding-bottom: 1rem !important;
  padding-left: 6rem !important;
}}
#MainMenu, footer, header {{ visibility: hidden; }}

/* ── ALL labels / inline text – force dark palette ── */
label,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li,
.stMarkdown p, .stMarkdown li, .stMarkdown span,
p, li {{
  color: var(--text) !important;
}}

/* ── input containers ── */
[data-baseweb="input"],
[data-baseweb="input"] > div,
[data-testid="stNumberInput"] input {{
  background: var(--surface-2) !important;
  color: var(--text) !important;
  border-color: var(--input-border) !important;
  box-shadow: none !important;
  outline: none !important;
}}
[data-baseweb="input"] {{
  border: 1px solid var(--input-border) !important;
  border-radius: 7px !important;
}}
[data-baseweb="input"] > div,
[data-testid="stNumberInput"] input {{
  border: 1px solid var(--input-border) !important;
  border-radius: 7px !important;
}}
[data-baseweb="input"]:hover > div,
[data-baseweb="input"]:focus-within > div,
[data-testid="stNumberInput"] input:hover,
[data-testid="stNumberInput"] input:focus {{
  border-color: color-mix(in srgb, var(--accent) 38%, var(--input-border)) !important;
  box-shadow: 0 0 0 1px color-mix(in srgb, var(--accent) 12%, transparent) !important;
}}
[data-baseweb="input"] input {{
  background: transparent !important;
  color: var(--text) !important;
  border: 0 !important;
  box-shadow: none !important;
  outline: none !important;
}}
[data-testid="stNumberInput"] [data-baseweb="input"],
[data-testid="stNumberInput"] [data-baseweb="input"] > div,
[data-testid="stNumberInput"] div:has(> input) {{
  background: var(--surface-2) !important;
  border-color: var(--input-border) !important;
  box-shadow: none !important;
  outline: none !important;
}}
[data-testid="stNumberInput"] * {{
  border-color: var(--input-border) !important;
}}
[data-testid="stNumberInput"] button {{
  background: var(--surface-2) !important;
  color: var(--muted) !important;
  border-color: var(--input-border) !important;
  box-shadow: none !important;
}}
[data-testid="stNumberInput"] button:hover {{
  background: color-mix(in srgb, var(--surface-2) 78%, var(--surface)) !important;
  color: var(--accent) !important;
  border-color: var(--input-border) !important;
}}
[data-testid="stNumberInput"] button svg {{
  color: var(--muted) !important;
  fill: var(--muted) !important;
}}

/* ── selectbox / dropdown ── */
[data-baseweb="select"] {{
  background: var(--surface-2) !important;
  background-color: var(--surface-2) !important;
  border-radius: 7px !important;
}}
[data-baseweb="select"] > div,
[data-baseweb="select"] > div:first-child {{
  min-height: 2.45rem;
  background: var(--surface-2) !important;
  background-color: var(--surface-2) !important;
  border: 1px solid var(--input-border) !important;
  border-radius: 7px !important;
  box-shadow: none !important;
  outline: none !important;
}}
[data-baseweb="select"] > div:hover,
[data-baseweb="select"] > div:focus-within {{
  border-color: color-mix(in srgb, var(--accent) 38%, var(--input-border)) !important;
  box-shadow: 0 0 0 1px color-mix(in srgb, var(--accent) 12%, transparent) !important;
}}
[data-baseweb="select"] input,
[data-baseweb="select"] span,
[data-baseweb="select"] div {{
  color: var(--text) !important;
}}
[data-baseweb="select"] > div *,
[data-baseweb="select"] [role="combobox"] {{
  background: var(--surface-2) !important;
  background-color: var(--surface-2) !important;
  border-color: transparent !important;
  box-shadow: none !important;
  outline: none !important;
}}
[data-baseweb="select"] span,
[data-baseweb="select"] svg {{
  background: transparent !important;
  background-color: transparent !important;
}}
[data-baseweb="select"] svg {{
  color: var(--muted) !important;
  fill: var(--muted) !important;
}}
[data-baseweb="select"] [aria-disabled="true"] {{
  color: var(--muted) !important;
}}

/* Streamlit renders the opened dropdown in a floating popover. */
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="menu"],
[role="listbox"] {{
  background: var(--surface-2) !important;
  background-color: var(--surface-2) !important;
  border-color: var(--input-border) !important;
  color: var(--text) !important;
}}
[data-baseweb="popover"] > div {{
  background: var(--surface-2) !important;
  background-color: var(--surface-2) !important;
  border: 1px solid var(--input-border) !important;
  border-radius: 7px !important;
  box-shadow: 0 16px 34px var(--shadow) !important;
  overflow: hidden !important;
}}
[data-baseweb="popover"] * {{
  background-color: var(--surface-2) !important;
  border-color: var(--input-border) !important;
  box-shadow: none !important;
}}
[data-baseweb="menu"],
[role="listbox"] {{
  padding: 4px !important;
}}
[data-baseweb="option"],
[role="option"] {{
  min-height: 2.2rem;
  background: var(--surface-2) !important;
  color: var(--text) !important;
  border-radius: 5px !important;
}}
[data-baseweb="option"]:hover,
[data-baseweb="option"][aria-selected="true"],
[role="option"]:hover,
[role="option"][aria-selected="true"] {{
  background: color-mix(in srgb, var(--accent) 18%, var(--surface)) !important;
  color: var(--accent) !important;
}}
[data-baseweb="option"] *,
[role="option"] * {{
  color: inherit !important;
}}

/* ── help tooltip bubbles ── */
[data-baseweb="tooltip"],
[data-testid="stTooltipContent"],
[role="tooltip"] {{
  background: var(--surface-2) !important;
  color: var(--text) !important;
  border: 1px solid var(--input-border) !important;
  border-radius: 7px !important;
  box-shadow: 0 14px 28px var(--shadow) !important;
}}
[data-baseweb="tooltip"] *,
[data-testid="stTooltipContent"] *,
[role="tooltip"] * {{
  background: transparent !important;
  color: var(--text) !important;
  border-color: var(--input-border) !important;
}}
[data-baseweb="tooltip"] svg,
[data-testid="stTooltipContent"] svg,
[role="tooltip"] svg {{
  color: var(--muted) !important;
  fill: var(--muted) !important;
}}
[data-testid="stTooltipIcon"] svg {{
  color: var(--muted) !important;
  fill: var(--muted) !important;
}}

/* ── file uploader ── */
[data-testid="stFileUploader"] {{
  background: transparent !important;
}}
[data-testid="stFileUploader"] section {{
  background: var(--surface-2) !important;
  border: 1px dashed var(--input-border) !important;
  border-radius: 8px !important;
  padding: 1rem !important;
}}
[data-testid="stFileUploader"] section * {{
  color: var(--text) !important;
}}
[data-testid="stFileUploader"] section small {{
  color: var(--muted) !important;
}}
[data-testid="stFileUploader"] button {{
  background: var(--surface) !important;
  border: 1px solid var(--input-border) !important;
  color: var(--text) !important;
  border-radius: 6px !important;
  box-shadow: none !important;
}}

/* ── element toolbar / fullscreen buttons ── */
[data-testid="stElementToolbar"],
[data-testid="stElementToolbar"] > div {{
  background: transparent !important;
  background-color: transparent !important;
  box-shadow: none !important;
}}
[data-testid="StyledFullScreenButton"],
[data-testid="stElementToolbar"] button,
[data-testid="stElementToolbar"] [role="button"],
button[title*="Full" i],
button[aria-label*="Full" i],
button[title*="expand" i],
button[aria-label*="expand" i] {{
  background: var(--accent) !important;
  background-color: var(--accent) !important;
  color: #fcfdfd !important;
  border: 1px solid color-mix(in srgb, var(--accent) 78%, #ffffff) !important;
  border-radius: 7px !important;
  box-shadow: 0 8px 18px color-mix(in srgb, var(--accent) 34%, transparent) !important;
  outline: none !important;
}}
[data-testid="StyledFullScreenButton"]::before,
[data-testid="StyledFullScreenButton"]::after,
[data-testid="stElementToolbar"] button::before,
[data-testid="stElementToolbar"] button::after,
button[title*="Full" i]::before,
button[title*="Full" i]::after,
button[aria-label*="Full" i]::before,
button[aria-label*="Full" i]::after {{
  background: transparent !important;
  background-color: transparent !important;
  box-shadow: none !important;
}}
[data-testid="StyledFullScreenButton"] *,
[data-testid="stElementToolbar"] button *,
[data-testid="stElementToolbar"] [role="button"] *,
button[title*="Full" i] *,
button[aria-label*="Full" i] *,
button[title*="expand" i] *,
button[aria-label*="expand" i] * {{
  background: transparent !important;
  background-color: transparent !important;
  box-shadow: none !important;
}}
[data-testid="StyledFullScreenButton"]:hover,
[data-testid="stElementToolbar"] button:hover,
[data-testid="stElementToolbar"] [role="button"]:hover,
button[title*="Full" i]:hover,
button[aria-label*="Full" i]:hover,
button[title*="expand" i]:hover,
button[aria-label*="expand" i]:hover {{
  background: color-mix(in srgb, var(--accent) 86%, #ffffff) !important;
  background-color: color-mix(in srgb, var(--accent) 86%, #ffffff) !important;
  color: #ffffff !important;
  border-color: color-mix(in srgb, var(--accent) 55%, #ffffff) !important;
  box-shadow: 0 10px 24px color-mix(in srgb, var(--accent) 46%, transparent) !important;
  filter: brightness(1.05);
}}
[data-testid="StyledFullScreenButton"] svg,
[data-testid="stElementToolbar"] svg,
button[title*="Full" i] svg,
button[aria-label*="Full" i] svg,
button[title*="expand" i] svg,
button[aria-label*="expand" i] svg {{
  color: inherit !important;
  fill: currentColor !important;
  stroke: currentColor !important;
}}

/* ── slider ── */
[data-testid="stSlider"] [role="slider"] {{
  background: var(--accent) !important;
}}
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {{
  color: var(--muted) !important;
}}

/* ── toggle ── */
[data-testid="stToggle"] p {{
  color: var(--text) !important;
}}

/* ── form container – strip default chrome ── */
[data-testid="stForm"] {{
  border: none !important;
  padding: 0 !important;
  background: transparent !important;
}}

/* ── alerts / info / warning / success – dark tint, low opacity ── */
div[data-testid="stAlert"] {{
  background: rgba(42, 37, 32, 0.55) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  backdrop-filter: blur(4px);
}}
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span {{
  color: var(--text) !important;
  opacity: 0.88;
}}
div[data-testid="stAlert"] svg {{
  fill: var(--muted) !important;
  opacity: 0.7;
}}

/* ── topbar ── */
.topbar {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  padding: 10px 16px;
  margin-bottom: 14px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: linear-gradient(135deg, var(--surface), var(--surface-2));
  box-shadow: 0 14px 36px var(--shadow);
}}
.brand-wrap {{
  display: flex;
  align-items: center;
  gap: 13px;
  min-width: 0;
}}
.logo-box {{
  width: 70px;
  height: 50px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 auto;
  overflow: hidden;
}}
.logo-box img {{
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}}
.brand {{
  display: flex;
  flex-direction: column;
  gap: 3px;
  min-width: 0;
}}
.brand strong {{ color: var(--text); font-size: 1.04rem; letter-spacing:.02em; line-height:1.15; }}
.brand span, .topbar .meta, .subtle {{ color: var(--muted); font-size:.78rem; }}
.brand span {{ line-height:1.25; }}
.topbar .meta {{ margin-left:auto; white-space:nowrap; }}

/* ── section title ── */
.section-title {{
  margin: 6px 0 10px;
  color: var(--text);
  font-size: .92rem;
  font-weight: 800;
  letter-spacing: .04em;
  text-transform: uppercase;
}}

/* ── panels / metric tiles ── */
.panel, .metric-tile, .model-box, .prediction-card {{
  border: 1px solid var(--border);
  border-radius: 8px;
  background: linear-gradient(180deg, var(--surface), var(--surface-2));
  box-shadow: 0 12px 28px var(--shadow);
}}
.panel {{ padding: 14px; margin-bottom: 14px; }}
.panel h3, .model-box h3 {{
  color: var(--accent);
  font-size: 1.2rem;
  font-weight: 800;
  letter-spacing: .08em;
  text-transform: uppercase;
}}
.metric-grid {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0,1fr));
  gap: 12px;
  margin-bottom: 14px;
}}
.metric-tile {{ min-height: 88px; padding: 12px; }}
.metric-tile .label {{ color:var(--muted); font-size:.74rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }}
.metric-tile .value {{ margin-top:8px; color:var(--text); font-size:1.55rem; font-weight:800; }}
.metric-tile .hint  {{ color:var(--muted); font-size:.75rem; }}
.model-box {{ padding:14px; }}

/* ── terminal ── */
.terminal-chrome {{
  background: #0a0a0a;
  border: 1px solid #2a2a2a;
  border-radius: 10px;
  overflow: hidden;
  font-family: 'JetBrains Mono','Courier New',monospace;
}}
.terminal-titlebar {{
  display: flex;
  align-items: center;
  gap: 6px;
  background: #1a1a1a;
  padding: 7px 12px;
  border-bottom: 1px solid #2a2a2a;
}}
.tdot {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
.tdot-r {{ background:#ff5f57; }}
.tdot-y {{ background:#febc2e; }}
.tdot-g {{ background:#28c840; }}
.terminal-title {{ color:#555; font-size:.70rem; margin-left:8px; }}
.terminal-body {{
  padding: 10px 14px 12px;
  overflow-y: auto;
  background: #0a0a0a;
}}
.tline {{
  font-size: .76rem;
  line-height: 1.72;
  color: #c9d1d9;
  white-space: pre-wrap;
}}
.tpfx {{ color:#3fb950; }}
.tnum {{ color:#444; user-select:none; }}
.tcursor {{
  display:inline-block; color:#3fb950;
  animation: tcblink 1s step-end infinite; font-size:.8rem;
}}
@keyframes tcblink {{ 0%,100%{{opacity:1}} 50%{{opacity:0}} }}

/* ── prediction card ── */
.prediction-card {{ padding:16px; text-align:center; }}
.prediction-card .grade {{ color:var(--accent); font-size:2.4rem; font-weight:800; }}
.prediction-card .confidence {{ color:var(--muted); font-size:.86rem; }}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {{
  gap:12px; padding:8px;
  justify-content:center;
  border:1px solid var(--border);
  border-radius:8px;
  background:var(--surface);
}}
.stTabs [data-baseweb="tab"] {{
  border:1px solid var(--border);
  border-radius:7px;
  background:color-mix(in srgb,var(--surface-2) 32%,var(--surface));
  color:var(--muted);
  font-weight:800; letter-spacing:.04em; padding:8px 14px;
}}
.stTabs [aria-selected="true"] {{
  background:var(--surface-2) !important;
  color:var(--accent) !important;
  border-color:var(--accent) !important;
}}

/* ── buttons ── */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
  border:0; border-radius:8px;
  background:var(--accent); color:#fcfdfd; font-weight:800;
}}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {{ filter:brightness(1.08); }}

/* ── data table ── */
.data-table-wrap {{
  width:100%; overflow:auto;
  border:1px solid var(--border); border-radius:8px;
  background:var(--surface);
  box-shadow:0 10px 24px var(--shadow);
}}
.data-table {{
  width:100%; min-width:760px;
  border-collapse:separate; border-spacing:0;
  color:var(--text); font-size:.82rem;
}}
.data-table thead th {{
  position:sticky; top:0; z-index:1;
  padding:10px 12px;
  background:var(--surface-2); color:var(--accent);
  border-bottom:1px solid var(--border);
  font-weight:800; text-align:left; white-space:nowrap;
}}
.data-table tbody td, .data-table tbody th {{
  padding:9px 12px;
  background:var(--surface); color:var(--text);
  border-bottom:1px solid color-mix(in srgb,var(--border) 65%,transparent);
  white-space:nowrap;
}}
.data-table tbody tr:nth-child(even) td,
.data-table tbody tr:nth-child(even) th {{
  background:color-mix(in srgb,var(--surface-2) 42%,var(--surface));
}}
.data-table tbody tr:hover td,
.data-table tbody tr:hover th {{
  background:color-mix(in srgb,var(--accent) 12%,var(--surface-2));
}}

.block-gap {{ height:18px; }}

@media (max-width:900px) {{
  .metric-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
  .topbar {{ align-items:flex-start; flex-direction:column; }}
  .brand-wrap {{ align-items:center; }}
  .topbar .meta {{ margin-left:0; white-space:normal; }}
}}
</style>
""", unsafe_allow_html=True)


# ── chart helpers ────────────────────────────────────────────────────────────
def chart_theme(ax: Any) -> None:
    ax.set_facecolor(CHART["panel"])
    ax.figure.set_facecolor(CHART["bg"])
    ax.tick_params(colors=CHART["muted"])
    ax.xaxis.label.set_color(CHART["text"])
    ax.yaxis.label.set_color(CHART["text"])
    ax.title.set_color(CHART["text"])
    ax.grid(True, color=CHART["grid"], alpha=0.32)
    for spine in ax.spines.values():
        spine.set_color(CHART["grid"])


def render_table(df: pd.DataFrame, height: int = 320, hide_index: bool = False) -> None:
    html = df.to_html(
        classes="data-table", border=0, index=not hide_index,
        escape=True, float_format=lambda v: f"{v:.4g}",
    )
    st.markdown(
        f'<div class="data-table-wrap" style="max-height:{height}px;">{html}</div>',
        unsafe_allow_html=True,
    )


def section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def metric_grid(items: list[tuple[str, str, str]], columns: int = 4) -> None:
    html = [f'<div class="metric-grid" style="grid-template-columns:repeat({columns},minmax(0,1fr));">']
    for label, value, hint in items:
        html.append(
            f'<div class="metric-tile"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="hint">{hint}</div></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


# ── terminal ─────────────────────────────────────────────────────────────────
def build_terminal_html(logs: list[str], title: str = "log",
                         done: bool = True, body_px: int = 260) -> str:
    lines = ""
    for i, line in enumerate(logs, 1):
        lines += (
            f'<div class="tline">'
            f'<span class="tnum">{i:02d} </span>'
            f'<span class="tpfx">▶ </span>{line}'
            f'</div>\n'
        )
    cursor = "" if done else '<div class="tline"><span class="tcursor">█</span></div>'
    uid = title.replace(" ", "_").replace(".", "_").replace("/", "_")
    return f"""
<div class="terminal-chrome">
  <div class="terminal-titlebar">
    <span class="tdot tdot-r"></span>
    <span class="tdot tdot-y"></span>
    <span class="tdot tdot-g"></span>
    <span class="terminal-title">{title}</span>
  </div>
  <div class="terminal-body" style="height:{body_px}px;" id="tb_{uid}">
{lines}{cursor}
  </div>
</div>
<script>
(function(){{var e=document.getElementById('tb_{uid}');if(e)e.scrollTop=e.scrollHeight;}})();
</script>
"""


# ── dataset charts ───────────────────────────────────────────────────────────
def plot_grade_distribution(df: pd.DataFrame, w=GRADE_W, h=GRADE_H) -> None:
    dist   = data_io.get_grade_distribution(df)
    colors = [CHART["accent_2"], CHART["accent_3"], CHART["accent"], "#9fa0b5", "#c6d1d7"]
    fig, ax = plt.subplots(figsize=(w, h))
    bars = ax.bar(dist["Grade"].astype(str), dist["Count"],
                  color=colors[:len(dist)], edgecolor=CHART["grid"])
    ax.set_title("GradeClass Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Students")
    ax.margins(x=0.08, y=0.14)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{int(bar.get_height())}", ha="center", va="bottom", color=CHART["text"])
    chart_theme(ax)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.84, bottom=0.13)
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def plot_grade_pie(df: pd.DataFrame, w=GRADE_W, h=GRADE_H) -> None:
    dist   = data_io.get_grade_distribution(df)
    colors = [CHART["accent_2"], CHART["accent_3"], CHART["accent"], "#9fa0b5", "#c6d1d7"]
    fig, ax = plt.subplots(figsize=(w, h))
    ax.pie(dist["Count"], labels=dist["Grade"].astype(str), autopct="%1.1f%%",
           startangle=90, colors=colors[:len(dist)], radius=0.88,
           textprops={"color": CHART["text"], "fontsize": 9},
           wedgeprops={"edgecolor": CHART["grid"], "linewidth": 1})
    ax.set_title("GradeClass Share")
    chart_theme(ax)
    ax.set_aspect("equal")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.84, bottom=0.08)
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def plot_missing_values(df: pd.DataFrame) -> None:
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(10)
    if missing.empty:
        st.info("No missing values found in the dataset.")
        return
    fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
    ax.barh(missing.index, missing.values, color=CHART["accent"], edgecolor=CHART["grid"])
    ax.set_title("Top Missing Value Columns"); ax.set_xlabel("Missing cells")
    chart_theme(ax); fig.tight_layout(); st.pyplot(fig, clear_figure=True)


def plot_relationships(df: pd.DataFrame) -> None:
    if "StudyTimeWeekly" not in df.columns or "GPA" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
    ax.scatter(df["StudyTimeWeekly"], df["GPA"], s=18, alpha=0.65,
               color=CHART["accent_2"], edgecolors="none", label="Study time")
    if "Absences" in df.columns:
        ax.scatter(df["Absences"], df["GPA"], s=18, alpha=0.45,
                   color=CHART["accent"], edgecolors="none", label="Absences")
    ax.set_title("Academic Patterns vs GPA")
    ax.set_xlabel("Hours or absences"); ax.set_ylabel("GPA")
    ax.legend(frameon=False, labelcolor=CHART["text"])
    chart_theme(ax); fig.tight_layout(); st.pyplot(fig, clear_figure=True)


# ── preprocessing charts (3-col, equal size) ─────────────────────────────────
def render_preprocess_charts(prep: Any) -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        fig, ax = plt.subplots(figsize=(PREP_W, PREP_H))
        prep.class_counts_before.sort_index().plot(
            kind="bar", ax=ax, color=CHART["accent_2"], edgecolor=CHART["grid"])
        ax.set_title("Before Balance"); ax.set_xlabel("Class"); ax.set_ylabel("Rows")
        chart_theme(ax); fig.tight_layout(); st.pyplot(fig, clear_figure=True)
    with c2:
        fig, ax = plt.subplots(figsize=(PREP_W, PREP_H))
        prep.class_counts_after.sort_index().plot(
            kind="bar", ax=ax, color=CHART["accent"], edgecolor=CHART["grid"])
        ax.set_title("After Balance"); ax.set_xlabel("Class")
        chart_theme(ax); fig.tight_layout(); st.pyplot(fig, clear_figure=True)
    with c3:
        split = pd.DataFrame({"Split": ["Train", "Test"],
                               "Rows":  [len(prep.X_train), len(prep.X_test)]})
        fig, ax = plt.subplots(figsize=(PREP_W, PREP_H))
        ax.pie(split["Rows"], labels=split["Split"], autopct="%1.0f%%",
               colors=[CHART["accent_2"], CHART["accent"]],
               textprops={"color": CHART["text"]},
               wedgeprops={"edgecolor": CHART["grid"]})
        ax.set_title("Train-Test Split")
        chart_theme(ax); fig.tight_layout(); st.pyplot(fig, clear_figure=True)


# ── evaluation charts (4-col, equal size) ────────────────────────────────────
def render_evaluation_charts(results: dict[str, dict[str, float]]) -> None:
    """Metric comparison + all confusion matrices in one equal-height row."""
    y_test      = st.session_state["preprocess_result"].y_test
    predictions = st.session_state["predictions"]
    labels      = st.session_state["preprocess_result"].metadata["class_labels"]

    n_cols = 1 + len(predictions)
    cols   = st.columns(n_cols)

    # col 0 → grouped bar chart
    with cols[0]:
        df_m = pd.DataFrame(results).T * 100
        fig, ax = plt.subplots(figsize=(EVAL_W, EVAL_H))
        df_m.plot(kind="bar", ax=ax,
                  color=[CHART["accent_2"], CHART["accent"], CHART["accent_3"], "#c6d1d7"],
                  edgecolor=CHART["grid"])
        ax.set_ylim(0, 105); ax.set_ylabel("Percent")
        ax.set_title("Metric Comparison")
        ax.set_xticklabels(df_m.index, rotation=0)
        ax.legend(frameon=False, labelcolor=CHART["text"], fontsize=7)
        chart_theme(ax); fig.tight_layout(); st.pyplot(fig, clear_figure=True)

    # remaining cols → confusion matrices
    for col_widget, (name, pred) in zip(cols[1:], predictions.items()):
        with col_widget:
            fig, ax = plt.subplots(figsize=(EVAL_W, EVAL_H))
            cm = confusion_matrix(y_test, pred, labels=labels)
            sns.heatmap(cm, annot=True, fmt="d", cbar=False, cmap="Oranges",
                        xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_title(f"{name} Confusion")
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            chart_theme(ax); fig.tight_layout(); st.pyplot(fig, clear_figure=True)


# ── ANN ──────────────────────────────────────────────────────────────────────
# ── training ──────────────────────────────────────────────────────────────────
# Training lives in train_models.py; app.py only calls the module from the UI.

def reset_downstream_state() -> None:
    for k in ["preprocess_result", "preprocess_log", "trained", "models",
              "results", "predictions", "ann_history", "training_log",
              "knn_curve", "svm_curve", "batch_output"]:
        st.session_state.pop(k, None)


def uploaded_file_signature(f: Any) -> tuple:
    return ("upload", getattr(f, "file_id", None), getattr(f, "name", ""),
            getattr(f, "size", None), getattr(f, "type", None))


# ════════════════════════════════════════════════════════════════════════════
# PAGE
# ════════════════════════════════════════════════════════════════════════════
inject_css()

logo_src = logo_data_uri(LOGO_PATH)

st.markdown(f"""
<div class="topbar">
  <div class="brand-wrap">
    <div class="logo-box">
      <img src="{logo_src}" alt="Student Performance BI System logo">
    </div>
    <div class="brand">
      <strong>Student Performance BI System</strong>
      <span>Dataset handling · Preprocessing · Training · Evaluation · Prediction</span>
    </div>
  </div>
  <div class="meta">KNN / SVM / ANN — GradeClass target</div>
</div>
""", unsafe_allow_html=True)

for key, val in [("df_raw", None), ("dataset_signature", None)]:
    if key not in st.session_state:
        st.session_state[key] = val

tabs = st.tabs(["Dataset", "Preprocessing", "Model Training", "Evaluation", "Prediction"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 0  ─  Dataset
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    section_title("Dataset")
    upload_col, overview_col = st.columns([0.85, 1.15], gap="medium")

    with upload_col:
        st.markdown('<div class="panel"><h3>Data Source</h3>'
                    '<div class="subtle">Upload CSV or Excel data.</div></div>',
                    unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Dataset file", type=["csv", "xlsx", "xls"],
            label_visibility="collapsed")

        if uploaded_file is not None:
            sig = uploaded_file_signature(uploaded_file)
            if st.session_state.dataset_signature != sig:
                if st.session_state.dataset_signature is None and st.session_state.df_raw is not None:
                    st.session_state.dataset_signature = sig
                    st.info("Dataset is already loaded.")
                else:
                    try:
                        uploaded_file.seek(0)
                        st.session_state.df_raw = data_io.load_dataset(uploaded_file)
                        st.session_state.dataset_signature = sig
                        reset_downstream_state()
                        st.success("Dataset loaded successfully.")
                    except Exception as exc:
                        st.error(f"Dataset loading failed: {exc}")

        sample_path = Path(__file__).with_name("synthetic_student_performance.csv")
        if uploaded_file is None and sample_path.exists():
            if st.button("Load Bundled Sample Dataset", use_container_width=True):
                st.session_state.df_raw = pd.read_csv(sample_path)
                st.session_state.dataset_signature = ("sample", str(sample_path))
                reset_downstream_state()
                st.success("Sample dataset loaded.")

    if st.session_state.df_raw is None:
        with upload_col:
            st.info("Load a dataset to preview rows, schema information, and overview charts.")
    else:
        df_raw = st.session_state.df_raw
        info   = data_io.get_dataset_info(df_raw)

        with upload_col:
            st.markdown('<div class="panel"><h3>Column Types</h3></div>',
                        unsafe_allow_html=True)
            render_table(data_io.get_dtype_table(df_raw), height=260, hide_index=True)
            st.markdown('<div class="block-gap"></div>', unsafe_allow_html=True)
            st.markdown('<div class="panel"><h3>Dataset Dictionary</h3></div>',
                        unsafe_allow_html=True)
            render_table(data_io.get_dictionary_table(), height=280, hide_index=True)
            st.markdown('<div class="block-gap"></div>', unsafe_allow_html=True)
            metric_grid([
                ("Records", f"{info.rows:,}",    "students"),
                ("Columns", f"{info.columns:,}", "dataset fields"),
                ("Missing", f"{info.missing_values:,}", "empty cells"),
                ("Target",  info.target_column,  "selected automatically"),
            ], columns=2)

        with overview_col:
            st.markdown('<div class="panel"><h3>Data Preview</h3></div>',
                        unsafe_allow_html=True)
            render_table(df_raw.head(40), height=360)
            plot_missing_values(df_raw)
            c_a, c_b = st.columns(2, gap="medium")
            with c_a:
                plot_grade_distribution(df_raw)
            with c_b:
                plot_grade_pie(df_raw)
            plot_relationships(df_raw)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1  ─  Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    section_title("Preprocessing")
    if st.session_state.df_raw is None:
        st.warning("Upload a dataset first in the Dataset tab.")
    else:
        opt_col, log_col = st.columns([1, 1], gap="medium")

        # ── right: terminal (always visible, empty until run) ──────────────
        with log_col:
            prep_term_ph = st.empty()
            prep_term_ph.markdown(
                build_terminal_html(
                    st.session_state.get("preprocess_log", []),
                    "preprocessing.log",
                    done=True,
                    body_px=260,
                ),
                unsafe_allow_html=True,
            )

        # ── left: options form ─────────────────────────────────────────────
        with opt_col:
            with st.form("preprocess_form"):
                balance_classes = st.toggle(
                    "Balance classes", value=True,
                    help="Undersamples majority classes so every class has equal rows. "
                         "Prevents bias toward the most common grade.",
                )
                scale_features = st.toggle(
                    "Feature scaling", value=True,
                    help="Applies StandardScaler (zero mean, unit variance). "
                         "Essential for KNN and SVM — without it, large-range features dominate.",
                )
                handle_missing = st.toggle(
                    "Handle missing values", value=True,
                    help="ON → numeric NaN filled with column median; "
                         "categorical NaN treated as 'Missing' category.\n\n"
                         "OFF → rows with any missing value are dropped entirely.",
                )
                encode_categorical = st.toggle(
                    "Encode categorical data", value=True,
                    help="ON → text/object columns are label-encoded to integers.\n\n"
                         "OFF → non-numeric columns are dropped before training. "
                         "Use only if your dataset is already fully numeric.",
                )
                test_size = st.slider(
                    "Test size", 0.1, 0.4, 0.2, 0.05,
                    help="Fraction of rows reserved for evaluation. "
                         "0.20 = 20 % test, 80 % train.",
                )
                preprocess_submitted = st.form_submit_button(
                    "Run Preprocess", use_container_width=True, type="primary")

        # ── run preprocessing on submit ────────────────────────────────────
        if preprocess_submitted:
            collected: list[str] = []

            def _stream_prep(line: str) -> None:
                collected.append(line)
                prep_term_ph.markdown(
                    build_terminal_html(collected, "preprocessing.log",
                                        done=False, body_px=260),
                    unsafe_allow_html=True,
                )
                time.sleep(0.04)

            try:
                result = pre_processing.preprocess_dataset(
                    st.session_state.df_raw,
                    test_size=test_size,
                    balance_classes=balance_classes,
                    scale_features=scale_features,
                    handle_missing=handle_missing,
                    encode_categorical=encode_categorical,
                    on_log=_stream_prep,
                )
                st.session_state.preprocess_result = result
                st.session_state.preprocess_log    = collected
                st.session_state.trained = False
                prep_term_ph.markdown(
                    build_terminal_html(collected, "preprocessing.log",
                                        done=True, body_px=260),
                    unsafe_allow_html=True,
                )
                st.success("Preprocessing completed.")
            except Exception as exc:
                st.error(f"Preprocessing failed: {exc}")

        # ── results below both columns ─────────────────────────────────────
        prep = st.session_state.get("preprocess_result")
        if prep is None:
            st.info("Click Run Preprocess to clean data, encode fields, "
                    "balance classes, scale features, and split train/test sets.")
        else:
            metric_grid([
                ("Raw Shape",     f"{prep.raw_shape[0]:,} × {prep.raw_shape[1]:,}", "before cleaning"),
                ("Cleaned Shape", f"{prep.cleaned_shape[0]:,} × {prep.cleaned_shape[1]:,}", "after balancing"),
                ("Features",      f"{len(prep.feature_columns):,}", "model inputs"),
                ("Target",        prep.target_column, "prediction output"),
            ])
            render_preprocess_charts(prep)
            split_df = pd.DataFrame({
                "Split": ["Training", "Testing"],
                "Rows":  [len(prep.X_train), len(prep.X_test)],
                "Classes": [prep.y_train.nunique(), prep.y_test.nunique()],
            })
            st.markdown('<div class="panel"><h3>Cleaned Feature Sample</h3></div>',
                        unsafe_allow_html=True)
            render_table(prep.X_train.head(12), height=320)
            render_table(split_df, height=180, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2  ─  Model Training
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    section_title("Model Training")
    prep = st.session_state.get("preprocess_result")
    if prep is None:
        st.warning("Run preprocessing before training models.")
    else:
        # 3 model cols (form) + 1 log col (always visible terminal)
        form_area, log_col = st.columns([3, 1], gap="medium")

        # ── right: training terminal (always visible) ──────────────────────
        with log_col:
            train_term_ph = st.empty()
            train_term_ph.markdown(
                build_terminal_html(
                    st.session_state.get("training_log", []),
                    "training.log",
                    done=True,
                    body_px=278,
                ),
                unsafe_allow_html=True,
            )

        # ── left: model params form ────────────────────────────────────────
        with form_area:
            with st.form("train_form"):
                p1, p2, p3 = st.columns(3, gap="medium")
                with p1:
                    st.markdown('<div class="model-box"><h3>KNN</h3>'
                                '<div class="subtle">Distance-based classifier.</div></div>',
                                unsafe_allow_html=True)
                    knn_k = st.number_input(
                        "k neighbors", 1, 51, 5, 2,
                        help="How many nearby students vote for the predicted class.")
                with p2:
                    st.markdown('<div class="model-box"><h3>SVM</h3>'
                                '<div class="subtle">Margin-based classifier.</div></div>',
                                unsafe_allow_html=True)
                    svm_kernel = st.selectbox(
                        "Kernel", ["rbf", "linear", "poly", "sigmoid"],
                        help="Shape of the decision boundary.")
                    svm_c = st.number_input(
                        "C regularization", 0.01, 100.0, 1.0, 0.1,
                        help="Higher C fits training data more tightly.")
                with p3:
                    st.markdown('<div class="model-box"><h3>ANN</h3>'
                                '<div class="subtle">Neural network classifier.</div></div>',
                                unsafe_allow_html=True)
                    ann_epochs = st.number_input(
                        "Epochs", 1, 200, 30, 1,
                        help="Training passes through the data.")
                    ann_batch = st.number_input(
                        "Batch size", 4, 256, 32, 4,
                        help="Rows used per training update.")

                if not TF_AVAILABLE:
                    st.warning(f"TensorFlow unavailable — ANN skipped. ({TF_IMPORT_ERROR})")

                train_submitted = st.form_submit_button(
                    "Train Models", use_container_width=True)

        # ── run training on submit ─────────────────────────────────────────
        if train_submitted:
            t_collected: list[str] = []

            def _stream_train(line: str) -> None:
                t_collected.append(line)
                train_term_ph.markdown(
                    build_terminal_html(t_collected, "training.log",
                                        done=False, body_px=278),
                    unsafe_allow_html=True,
                )
                time.sleep(0.03)

            try:
                params = {
                    "knn_k":      int(knn_k),
                    "svm_kernel": svm_kernel,
                    "svm_c":      float(svm_c),
                    "ann_epochs": int(ann_epochs),
                    "ann_batch":  int(ann_batch),
                }
                (models, results, predictions,
                 ann_history, training_log,
                 knn_curve, svm_curve) = model_training.train_models(
                    prep, params, on_log=_stream_train)

                st.session_state.update({
                    "models": models, "results": results,
                    "predictions": predictions, "ann_history": ann_history,
                    "training_log": training_log,
                    "knn_curve": knn_curve, "svm_curve": svm_curve,
                    "trained": True,
                })
                train_term_ph.markdown(
                    build_terminal_html(t_collected, "training.log",
                                        done=True, body_px=278),
                    unsafe_allow_html=True,
                )
                st.success("Models trained successfully.")
            except Exception as exc:
                t_collected.append(f"ERROR: {exc}")
                train_term_ph.markdown(
                    build_terminal_html(t_collected, "training.log",
                                        done=True, body_px=278),
                    unsafe_allow_html=True,
                )
                st.error(f"Training failed: {exc}")

        # ── progress charts (shown after training) ─────────────────────────
        if st.session_state.get("trained"):
            knn_curve = st.session_state.get("knn_curve")
            svm_curve = st.session_state.get("svm_curve")
            history   = st.session_state.get("ann_history")

            if knn_curve or svm_curve or history:
                section_title("Training Progress Charts")

            if knn_curve and svm_curve:
                ck, cs = st.columns(2)
                with ck:
                    fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
                    ax.plot(knn_curve["k_range"], knn_curve["train"],
                            color=CHART["accent_2"], marker="o", markersize=4, label="Train")
                    ax.plot(knn_curve["k_range"], knn_curve["test"],
                            color=CHART["accent"],   marker="o", markersize=4, label="Test")
                    ax.set_title("KNN — Accuracy vs k")
                    ax.set_xlabel("k"); ax.set_ylabel("Accuracy")
                    ax.legend(frameon=False, labelcolor=CHART["text"])
                    chart_theme(ax); fig.tight_layout(); st.pyplot(fig, clear_figure=True)
                with cs:
                    fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
                    ax.plot(svm_curve["sizes"], svm_curve["train"],
                            color=CHART["accent_2"], marker="o", markersize=4, label="Train")
                    ax.plot(svm_curve["sizes"], svm_curve["test"],
                            color=CHART["accent"],   marker="o", markersize=4, label="Test")
                    ax.set_title("SVM — Learning Curve")
                    ax.set_xlabel("Training samples"); ax.set_ylabel("Accuracy")
                    ax.legend(frameon=False, labelcolor=CHART["text"])
                    chart_theme(ax); fig.tight_layout(); st.pyplot(fig, clear_figure=True)

            if history is not None:
                epochs = range(1, len(history.history["loss"]) + 1)
                ca, cl = st.columns(2)
                with ca:
                    fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
                    ax.plot(epochs, history.history["accuracy"],     color=CHART["accent_2"], label="Train")
                    ax.plot(epochs, history.history["val_accuracy"], color=CHART["accent"],   label="Validation")
                    ax.set_title("ANN — Accuracy"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
                    ax.legend(frameon=False, labelcolor=CHART["text"])
                    chart_theme(ax); fig.tight_layout(); st.pyplot(fig, clear_figure=True)
                with cl:
                    fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
                    ax.plot(epochs, history.history["loss"],     color=CHART["accent_2"], label="Train")
                    ax.plot(epochs, history.history["val_loss"], color=CHART["accent"],   label="Validation")
                    ax.set_title("ANN — Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
                    ax.legend(frameon=False, labelcolor=CHART["text"])
                    chart_theme(ax); fig.tight_layout(); st.pyplot(fig, clear_figure=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3  ─  Evaluation
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    section_title("Evaluation")
    if not st.session_state.get("trained"):
        st.warning("Train models before viewing evaluation results.")
    else:
        results    = st.session_state.results
        best_model = max(results, key=lambda n: results[n]["Accuracy"])
        metric_grid([
            ("Best Model", best_model,                                    "highest accuracy"),
            ("Accuracy",   f"{results[best_model]['Accuracy']*100:.2f}%", "holdout test set"),
            ("F1 Score",   f"{results[best_model]['F1 Score']*100:.2f}%", "weighted average"),
            ("Models",     f"{len(results)}",                             "trained"),
        ])
        st.markdown('<div class="panel"><h3>Model Metrics Table</h3></div>',
                    unsafe_allow_html=True)
        eval_df = pd.DataFrame(results).T.rename(columns={"F1 Score": "F1-score"}).round(4)
        render_table(eval_df, height=240)
        # 4-chart row: metric comparison + confusion matrices
        render_evaluation_charts(results)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4  ─  Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    section_title("Prediction")
    if not st.session_state.get("trained"):
        st.warning("Train at least one model before making predictions.")
    else:
        models = st.session_state.models
        prep   = st.session_state.preprocess_result
        single_tab, batch_tab = st.tabs(["Single Prediction", "Batch Prediction"])

        # ── single prediction (wrapped in form → no rerun on widget change) ──
        with single_tab:
            with st.form("single_pred_form"):
                model_choice = st.selectbox(
                    "Choose model", list(models.keys()),
                    help="Which trained model generates the prediction.")
                st.markdown('<div class="panel"><h3>Student Features</h3></div>',
                            unsafe_allow_html=True)
                i1, i2, i3 = st.columns(3)
                with i1:
                    age            = st.number_input("Age", 15, 18, 17)
                    gender_label   = st.selectbox("Gender",
                        list(predictor.CATEGORY_OPTIONS["Gender"].keys()))
                    ethnicity_label = st.selectbox("Ethnicity",
                        list(predictor.CATEGORY_OPTIONS["Ethnicity"].keys()))
                    education_label = st.selectbox("Parental education",
                        list(predictor.CATEGORY_OPTIONS["ParentalEducation"].keys()))
                with i2:
                    study          = st.number_input("Study time weekly", 0.0, 20.0, 10.0, 0.5)
                    absences       = st.number_input("Absences", 0, 30, 5)
                    tutoring_label = st.selectbox("Tutoring",
                        list(predictor.CATEGORY_OPTIONS["Tutoring"].keys()))
                    support_label  = st.selectbox("Parental support",
                        list(predictor.CATEGORY_OPTIONS["ParentalSupport"].keys()))
                with i3:
                    extra_label        = st.selectbox("Extracurricular",
                        list(predictor.CATEGORY_OPTIONS["Extracurricular"].keys()))
                    sports_label       = st.selectbox("Sports",
                        list(predictor.CATEGORY_OPTIONS["Sports"].keys()))
                    music_label        = st.selectbox("Music",
                        list(predictor.CATEGORY_OPTIONS["Music"].keys()))
                    volunteering_label = st.selectbox("Volunteering",
                        list(predictor.CATEGORY_OPTIONS["Volunteering"].keys()))
                    gpa = st.number_input("GPA", 2.0, 4.0, 3.0, 0.05)

                predict_submitted = st.form_submit_button(
                    "Predict Student Grade", use_container_width=True)

            if predict_submitted:
                input_row = {
                    "Age":               age,
                    "Gender":            predictor.CATEGORY_OPTIONS["Gender"][gender_label],
                    "Ethnicity":         predictor.CATEGORY_OPTIONS["Ethnicity"][ethnicity_label],
                    "ParentalEducation": predictor.CATEGORY_OPTIONS["ParentalEducation"][education_label],
                    "StudyTimeWeekly":   study,
                    "Absences":          absences,
                    "Tutoring":          predictor.CATEGORY_OPTIONS["Tutoring"][tutoring_label],
                    "ParentalSupport":   predictor.CATEGORY_OPTIONS["ParentalSupport"][support_label],
                    "Extracurricular":   predictor.CATEGORY_OPTIONS["Extracurricular"][extra_label],
                    "Sports":            predictor.CATEGORY_OPTIONS["Sports"][sports_label],
                    "Music":             predictor.CATEGORY_OPTIONS["Music"][music_label],
                    "Volunteering":      predictor.CATEGORY_OPTIONS["Volunteering"][volunteering_label],
                    "GPA":               gpa,
                }
                input_row = {k: v for k, v in input_row.items()
                             if k in prep.metadata["feature_columns"]}
                result     = predictor.predict_single(
                    input_row, models[model_choice], model_choice, prep.metadata)
                confidence = ("Not available" if result["confidence"] is None
                              else f"{result['confidence']*100:.2f}%")
                st.markdown(f"""
<div class="prediction-card">
  <div class="subtle">{model_choice} predicted grade</div>
  <div class="grade">{result["label"]}</div>
  <div class="confidence">Confidence: {confidence}</div>
</div>
""", unsafe_allow_html=True)

        # ── batch prediction (uses already-loaded dataset) ─────────────────
        with batch_tab:
            with st.form("batch_pred_form"):
                batch_model_choice = st.selectbox(
                    "Choose model for batch prediction", list(models.keys()),
                    help="Which trained model processes the batch.")
                batch_submitted = st.form_submit_button(
                    "Run Batch Prediction", use_container_width=True)

            df_raw = st.session_state.get("df_raw")
            if df_raw is None:
                st.warning("No dataset loaded. Go to the Dataset tab first.")
            else:
                st.markdown(
                    f'<div class="panel"><h3>Batch Prediction</h3>'
                    f'<div class="subtle">Using the dataset loaded in the Dataset tab — '
                    f'{len(df_raw):,} rows.</div></div>',
                    unsafe_allow_html=True,
                )
                if batch_submitted:
                    try:
                        batch_output = predictor.batch_predict(
                            df_raw, models[batch_model_choice],
                            batch_model_choice, prep.metadata)
                        st.session_state["batch_output"] = batch_output
                        st.success(f"Batch prediction complete — "
                                   f"{len(batch_output):,} rows processed.")
                    except Exception as exc:
                        st.error(f"Batch prediction failed: {exc}")

                batch_output = st.session_state.get("batch_output")
                if batch_output is not None:
                    render_table(batch_output.head(100), height=420)
                    st.download_button(
                        "Download Batch Predictions",
                        data=batch_output.to_csv(index=False).encode("utf-8"),
                        file_name="batch_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
