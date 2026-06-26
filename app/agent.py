import os
import re
from difflib import SequenceMatcher
from typing import Optional

import pandas as pd

try:
    import openai
except ImportError:
    openai = None

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from app.logging_config import logger
from app.data_loader import load_all_datasets
from app.chains import build_reasoning_chain
from app.intent import Intent, detect_intent
from app.utils import get_top_countries
from app.dataset_state import get_processed_path, get_active_source

from app.data_loader import (
    DataFileError,
    load_country_trends,
    load_processed_data,
    load_regional_trends,
)
from app.weather import (
    extract_location,
    extract_historical_date,
    is_historical_weather_question,
    format_weather_context,
    get_current_weather,
    get_historical_weather,
    normalize_weather,
    normalize_location,
) 

load_dotenv()

# =========================
# RESPONSE FORMATTER — inlined (output only, no logic change)
# =========================
# All formatter code lives here so this is a single self-contained file.

import re


# ── colour tokens ─────────────────────────────────────────────────────────

_CSS = """<style>
.el{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
    font-size:14px;line-height:1.75;color:#1f2937;max-width:100%}
.el-intro{margin:0 0 14px;color:#374151;font-size:14px}

/* section heading */
.el-h{font-size:11px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;
       color:#6b7280;margin:20px 0 8px;padding-bottom:5px;
       border-bottom:1px solid #e5e7eb}

/* stat cards */
.el-cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:8px;margin:10px 0}
.el-card{background:#f9fafb;border:1px solid #e5e7eb;border-radius:10px;padding:11px 13px}
.el-card-label{font-size:10px;color:#9ca3af;font-weight:600;text-transform:uppercase;
                letter-spacing:.05em;margin-bottom:4px}
.el-card-value{font-size:20px;font-weight:700;line-height:1.1;color:#111827}
.el-card-value.hot{color:#dc2626}.el-card-value.cold{color:#2563eb}
.el-card-value.up{color:#16a34a}.el-card-value.down{color:#ca8a04}

/* table */
.el-tw{overflow-x:auto;margin:10px 0;border-radius:10px;border:1px solid #e5e7eb}
.el-t{width:100%;border-collapse:collapse;font-size:13px}
.el-t th{background:#f3f4f6;color:#374151;font-weight:600;padding:8px 13px;
          text-align:left;border-bottom:1px solid #e5e7eb;white-space:nowrap}
.el-t td{padding:7px 13px;border-bottom:1px solid #f3f4f6;color:#1f2937;vertical-align:middle}
.el-t tr:last-child td{border-bottom:none}
.el-t tr:hover td{background:#fafafa}
.el-rn{color:#9ca3af;font-weight:700;font-size:11px;width:26px;
        display:inline-block;text-align:right;margin-right:5px}

/* bar */
.el-bar{display:flex;align-items:center;gap:8px}
.el-bar-bg{flex:1;height:5px;background:#e5e7eb;border-radius:99px;
            overflow:hidden;min-width:50px}
.el-bar-fill{height:5px;border-radius:99px}
.el-bar-fill.hot{background:#ef4444}.el-bar-fill.cold{background:#3b82f6}
.el-bar-fill.up{background:#22c55e}.el-bar-fill.neu{background:#6b7280}

/* badge */
.el-badge{display:inline-block;font-size:11px;font-weight:600;padding:2px 8px;
           border-radius:99px;white-space:nowrap;margin:0 2px}
.el-badge.hot{background:#fee2e2;color:#991b1b;border:1px solid #fecaca}
.el-badge.cold{background:#dbeafe;color:#1e40af;border:1px solid #bfdbfe}
.el-badge.up{background:#dcfce7;color:#166534;border:1px solid #bbf7d0}
.el-badge.down{background:#fef9c3;color:#854d0e;border:1px solid #fef08a}
.el-badge.neu{background:#f3f4f6;color:#374151;border:1px solid #e5e7eb}

/* note / callout */
.el-note{padding:10px 14px;border-radius:0 8px 8px 0;margin:10px 0;
          font-size:13px;border-left:3px solid}
.el-note.info{background:#eff6ff;border-color:#3b82f6;color:#1d4ed8}
.el-note.warn{background:#fff7ed;border-color:#f97316;color:#9a3412}
.el-note.good{background:#f0fdf4;border-color:#22c55e;color:#166534}
.el-note.neu{background:#f9fafb;border-color:#9ca3af;color:#374151}

/* insight list */
.el-insights{margin:10px 0;display:flex;flex-direction:column;gap:5px}
.el-insight{display:flex;gap:9px;align-items:flex-start;background:#f9fafb;
             border:1px solid #e5e7eb;border-radius:8px;padding:9px 12px;font-size:13px}
.el-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;margin-top:5px}

/* compare cards */
.el-cmp{display:grid;gap:10px;margin:10px 0}
.el-cmp-card{background:#f9fafb;border:1px solid #e5e7eb;border-radius:10px;padding:12px 15px}
.el-cmp-title{font-weight:700;font-size:14px;margin-bottom:8px;padding-bottom:6px;
               border-bottom:1px solid #e5e7eb;color:#111827}
.el-cmp-row{display:flex;justify-content:space-between;align-items:center;
             padding:4px 0;font-size:12px;border-bottom:1px solid #f3f4f6}
.el-cmp-row:last-child{border-bottom:none}
.el-cmp-key{color:#6b7280}.el-cmp-val{font-weight:600;color:#111827}

/* weather card */
.el-wx{background:linear-gradient(135deg,#1e3a5f,#1d4ed8);border-radius:14px;
        padding:20px;color:#fff;margin:8px 0}
.el-wx-tmp{font-size:52px;font-weight:800;line-height:1;margin-bottom:3px}
.el-wx-cond{font-size:14px;opacity:.85;margin-bottom:14px}
.el-wx-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px}
.el-wx-cell{background:rgba(255,255,255,.13);border-radius:8px;padding:8px 11px}
.el-wx-lbl{font-size:10px;opacity:.65;text-transform:uppercase;letter-spacing:.06em;margin-bottom:2px}
.el-wx-val{font-size:15px;font-weight:700}

/* prose */
.el-p{margin:0 0 10px;color:#374151;font-size:13px}
.el-ul{margin:6px 0 10px 18px;padding:0}
.el-ul li{margin:3px 0;font-size:13px;color:#374151}

/* source */
.el-src{font-size:11px;color:#9ca3af;margin-top:14px;padding-top:10px;
         border-top:1px solid #f3f4f6;display:flex;align-items:center;gap:5px}
</style>"""


# ── helpers ─────────────────────────────────────────────────────────────

def _x(t):
    return str(t).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def _color_class(val_str):
    """Guess a colour class from a value string."""
    s = str(val_str).strip()
    try:
        v = float(re.sub(r"[^\d.\-]","",s.split()[0]))
        if re.search(r"change|trend|warming|diff",s,re.I):
            return "up" if v>=0 else "down"
        if v>25: return "hot"
        if v<5:  return "cold"
    except Exception:
        pass
    return "neu"



_DOT_COLOR = {
    "hot":"#ef4444","cold":"#3b82f6","up":"#22c55e",
    "down":"#ca8a04","warn":"#f97316","info":"#3b82f6","neu":"#9ca3af",
}


# ── markdown → HTML ────────────────────────────────────────────────────

def _md_inline(text):
    """Convert **bold**, *italic* to HTML inside a string."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*",     r"<em>\1</em>",         text)
    return text

def _render_bullets(lines):
    """Convert a run of '- ...' lines into a <ul>."""
    items = "".join(f"<li>{_md_inline(_x(l))}</li>" for l in lines)
    return f'<ul class="el-ul">{items}</ul>'

def _render_prose(text):
    """Turn a plain paragraph (with possible inline **) into <p>."""
    text = text.strip()
    if not text:
        return ""
    return f'<p class="el-p">{_md_inline(_x(text))}</p>'


_LABEL_VALUE_RE = re.compile(
    r"^\s*(?:\*\*)?([^:*]+?)(?:\*\*)?\s*[:\-]\s*(.+?)\s*$"
)


def _parse_label_value_line(line):
    """Return (label, value) for Markdown/plain label-value lines."""
    m = _LABEL_VALUE_RE.match(str(line).strip())
    if not m:
        return None
    label = m.group(1).strip()
    value = m.group(2).strip()
    if not label or not value:
        return None
    return label, value


def _label_key(label):
    return re.sub(r"[^a-z0-9]+", "", str(label).lower())


def _render_stat_cards(cards):
    """Render consecutive parsed label-value pairs as one card grid."""
    if not cards:
        return ""
    inner = []
    for label, value in cards:
        cls = _color_class(value)
        inner.append(
            f'<div class="el-card">'
            f'<div class="el-card-label">{_x(label)}</div>'
            f'<div class="el-card-value {cls}">{_x(value)}</div>'
            f'</div>'
        )
    return f'<div class="el-cards">{"".join(inner)}</div>'


def _append_unit(value, unit):
    """Append a display unit only when the value does not already include it."""
    value = str(value).strip()
    if not value or value == "-":
        return "-"
    normalized = value.lower().replace(" ", "")
    unit_normalized = unit.lower().replace(" ", "")
    if unit_normalized in normalized:
        return value
    return f"{value}{unit}"


# ── table renderer ──────────────────────────────────────────────────────

def _render_table(rows, is_ranked=False):
    """
    rows: list of lists (first row = headers if is_ranked=False and no '#' col)
    is_ranked: add rank numbers + bar chart on last numeric column
    """
    if not rows:
        return ""
    headers = rows[0]
    body    = rows[1:]
    if not body:
        return ""

    # find numeric column for bar
    bar_idx, bar_max = None, 1.0
    if is_ranked:
        for ci in range(len(headers)-1, -1, -1):
            try:
                vals = [float(re.sub(r"[^\d.\-]","",r[ci])) for r in body if ci<len(r)]
                if vals:
                    bar_max = max(abs(v) for v in vals) or 1.0
                    bar_idx = ci
                    break
            except Exception:
                pass

    head_html = "".join(f"<th>{_x(h)}</th>" for h in headers)
    rows_html = []
    for ri, row in enumerate(body):
        cells = []
        for ci, cell in enumerate(row):
            cell_s = str(cell).strip()
            if ci == 0 and is_ranked:
                cells.append(f'<td><span class="el-rn">#{ri+1}</span>{_x(cell_s)}</td>')
            elif ci == bar_idx and is_ranked:
                try:
                    v   = float(re.sub(r"[^\d.\-]","",cell_s))
                    pct = min(100, abs(v)/bar_max*100)
                    cls = _color_class(cell_s)
                    cells.append(
                        f'<td><div class="el-bar">'
                        f'<span style="min-width:52px;font-weight:600">{_x(cell_s)}</span>'
                        f'<div class="el-bar-bg"><div class="el-bar-fill {cls}" style="width:{pct:.1f}%"></div></div>'
                        f'</div></td>'
                    )
                except Exception:
                    cells.append(f"<td>{_x(cell_s)}</td>")
            else:
                cells.append(f"<td>{_x(cell_s)}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    return (
        f'<div class="el-tw"><table class="el-t">'
        f"<thead><tr>{head_html}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        f"</table></div>"
    )


# ── main block parser ───────────────────────────────────────────────────

def _parse_blocks(text):
    """
    Walk the LLM output line-by-line and emit structured HTML blocks.
    Consumes each parsed line exactly once so label-value fields cannot be
    rendered by multiple block renderers.
    """
    lines  = text.split("\n")
    out    = []
    buf_bullets = []
    buf_table   = []   # list of row-lists
    buf_para    = []
    buf_cards   = []

    rank_sections = {
        "rank","top","hottest","coldest","warmest","coolest",
        "fastest","slowest","highest","lowest","warming",
    }

    current_section_is_rank = False

    def flush_cards():
        if buf_cards:
            out.append(_render_stat_cards(buf_cards))
            buf_cards.clear()

    def flush_bullets():
        if buf_bullets:
            flush_cards()
            out.append(_render_bullets(buf_bullets))
            buf_bullets.clear()

    def flush_table():
        if buf_table:
            flush_cards()
            out.append(_render_table(buf_table, is_ranked=current_section_is_rank))
            buf_table.clear()

    def flush_para():
        if buf_para:
            flush_cards()
            para = " ".join(buf_para).strip()
            if para:
                out.append(_render_prose(para))
            buf_para.clear()

    i = 0
    while i < len(lines):
        raw  = lines[i]
        line = raw.strip()
        i   += 1

        if not line:
            flush_bullets()
            flush_table()
            flush_para()
            flush_cards()
            continue

        if line.startswith("##"):
            flush_bullets(); flush_table(); flush_para(); flush_cards()
            title = re.sub(r"^#+\s*", "", line).strip()
            current_section_is_rank = any(w in title.lower() for w in rank_sections)
            out.append(f'<div class="el-h">{_x(title)}</div>')
            continue

        if line.startswith("|"):
            flush_bullets(); flush_para(); flush_cards()
            if not re.match(r"^[\|\s\-:]+$", line):
                cells = [c.strip() for c in line.strip("|").split("|")]
                buf_table.append(cells)
            continue

        if buf_table and not line.startswith("|"):
            flush_table()

        if re.match(r"^[-•*]\s+", line):
            flush_table(); flush_para(); flush_cards()
            content = re.sub(r"^[-•*]\s+", "", line)
            buf_bullets.append(content)
            continue

        if buf_bullets and not re.match(r"^[-•*]\s+", line):
            flush_bullets()

        if re.match(r"^\d+[\.)]\s+", line):
            flush_table(); flush_para(); flush_cards()
            content = re.sub(r"^\d+[\.)]\s+", "", line)
            buf_bullets.append(content)
            continue

        label_value = _parse_label_value_line(line)
        if label_value:
            flush_bullets(); flush_table(); flush_para()
            buf_cards.append(label_value)
            continue

        if re.match(r"^source\s*:", line, re.I):
            flush_bullets(); flush_table(); flush_para(); flush_cards()
            continue

        flush_table(); flush_cards()
        buf_para.append(line)

    flush_bullets()
    flush_table()
    flush_para()
    flush_cards()

    return "".join(out)


# group consecutive stat cards into a grid
def _merge_stat_grids(html):
    """
    Legacy compatibility hook. Stat cards are grouped during parsing now,
    avoiding fragile regex over nested divs that could drop card values.
    """
    return html


# weather card builder
def _weather_card(text):
    """
    If the response is a weather answer, extract fields and render a card.
    Returns (card_html, remaining_text). Weather label-value lines are consumed
    here so the generic stat-card parser cannot render them a second time.
    """
    weather_keys = {
        "location": "loc",
        "temperature": "temp",
        "feelslike": "feels",
        "condition": "cond",
        "humidity": "hum",
        "windspeed": "wind",
        "wind": "wind",
        "todaymin": "mn",
        "min": "mn",
        "todaymax": "mx",
        "max": "mx",
        "rain": "rain",
        "rainfall": "rain",
    }
    fields = {}
    consumed = set()
    lines = text.split("\n")

    for idx, raw in enumerate(lines):
        parsed = _parse_label_value_line(raw)
        if not parsed:
            continue
        label, value = parsed
        key = weather_keys.get(_label_key(label))
        if key and value:
            fields[key] = value
            consumed.add(idx)

    if "loc" not in fields:
        heading = next((l.strip() for l in lines if l.strip().lower().startswith("## weather")), "")
        if heading:
            fields["loc"] = re.sub(r"^#+\s*weather\s*[—-]?\s*", "", heading, flags=re.I).strip() or "Weather"

    has_weather_payload = any(k in fields for k in ["temp", "feels", "cond", "hum", "wind", "mn", "mx", "rain"])
    if not has_weather_payload:
        return "", text

    loc = fields.get("loc", "Weather")
    temp = _append_unit(fields.get("temp", "-"), " C")
    feels = _append_unit(fields.get("feels", "-"), " C")
    cond = fields.get("cond", "-")
    hum = fields.get("hum", "-")
    wind = _append_unit(fields.get("wind", "-"), " km/h")
    mn = _append_unit(fields.get("mn", "-"), " C")
    mx = _append_unit(fields.get("mx", "-"), " C")
    rain = _append_unit(fields.get("rain", "-"), " mm")

    cells = []
    for label, value in [
        ("Feels Like", feels),
        ("Humidity", hum),
        ("Condition", cond),
        ("Wind", wind),
        ("Min", mn),
        ("Max", mx),
    ]:
        if value != "-":
            cells.append(
                f'<div class="el-wx-cell"><div class="el-wx-lbl">{_x(label)}</div>'
                f'<div class="el-wx-val">{_x(value)}</div></div>'
            )

    card = (
        f'<div class="el-wx">'
        f'<div style="font-size:11px;opacity:.65;margin-bottom:6px;'
        f'text-transform:uppercase;letter-spacing:.07em">{_x(loc)}</div>'
        f'<div class="el-wx-tmp">{_x(temp)}</div>'
        f'<div class="el-wx-cond">{_x(cond)}'
        f'{f" · Feels like {_x(feels)}" if feels != "-" else ""}</div>'
        f'<div class="el-wx-grid">{"".join(cells)}</div>'
        f'<div style="margin-top:10px;font-size:12px;opacity:.65">Rainfall: {_x(rain)}</div>'
        f'</div>'
    )

    cleaned_lines = [raw for idx, raw in enumerate(lines) if idx not in consumed]
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"^Using live weather data,?\s*\n?", "", cleaned, flags=re.I).strip()

    return card, cleaned

# ── public API ───────────────────────────────────────────────────────────

def format_response(response_text: str, source: str = "") -> str:
    """
    Convert an EcoLens AI response string into rich, structured HTML.

    Parameters
    ----------
    response_text : str   Raw text returned by ask_ecolens()
    source        : str   Source string for the footer

    Returns
    -------
    str   Self-contained HTML (includes <style>)
    """
    if not response_text or not response_text.strip():
        return f"{_CSS}<div class='el'><p class='el-p'>No response available.</p></div>"

    # Strip trailing "Source: ..." line — we add a styled footer
    resolved_source = source
    src_m = re.search(r"\n*Source:\s*(.+)$", response_text, re.I)
    if src_m:
        resolved_source = src_m.group(1).strip()
    text = re.sub(r"\n*Source:\s*.+$", "", response_text, flags=re.I).strip()

    # ── weather card (full-bleed, rendered first) ──────────────────────
    wx_card, text = _weather_card(text)

    # ── strip the "Using live weather data," opener ─────────────────────
    text = re.sub(r"^Using live weather data,?\s*\n?", "", text, flags=re.I).strip()

    # ── parse remaining markdown into HTML blocks ───────────────────────
    body = _parse_blocks(text)

    # ── merge consecutive single stat-cards into grids ──────────────────
    body = _merge_stat_grids(body)

    # ── source footer ────────────────────────────────────────────────────
    footer = (
        f'<div class="el-src">'
        f'<span>Source: {_x(resolved_source)}</span>'
        f'</div>'
    )
    inner = wx_card + body + footer
    return f"{_CSS}<div class='el'>{inner}</div>"


# Bind the public name used throughout this file
_fmt_response = format_response
_FMT = True

# =========================
# CONSTANTS
# =========================

CLIMATE_SOURCE = "Historical Climate Dataset"
WEATHER_SOURCE = "Live Weather"
DOMAIN_REJECTION = "EcoLens only supports weather and climate related questions."
CLIMATE_UNAVAILABLE = "Climate dataset temporarily unavailable."
WEATHER_UNAVAILABLE = "Live weather service unavailable."

# Single source of truth for history window
HISTORY_WINDOW = 10  # FIX-6: increased from 6 — no practical chat limit issue

# Context size caps to prevent token overflow
MAX_FACTS_CHARS = 14_000   # slightly larger for ranking/compare tables
MAX_EVIDENCE_CHARS = 3_000
MAX_SUPPLEMENT_CHARS = 5_000
MAX_PRIOR_CONTEXT_CHARS = 800   # FIX-1: tightened — only for genuine follow-ups
MAX_LAST_ANSWER_CHARS = 1_500   # FIX-1: tightened
MAX_LAST_EVIDENCE_CHARS = 4_000


# =========================
# OPENAI CONFIG
# =========================

class OpenAIConfigError(Exception):
    pass

class GeminiConfigError(Exception):
    pass


def validate_openai_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise OpenAIConfigError("OPENAI_API_KEY is missing in .env file")
    if not api_key.startswith("sk-"):
        raise OpenAIConfigError("OPENAI_API_KEY looks invalid")

def validate_gemini_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise GeminiConfigError("GEMINI_API_KEY is missing in .env file")
    return True

def get_llm():
    validate_gemini_key()
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        timeout=60,
        max_retries=2,
    )


def clean_history(history):
    """Normalize and trim conversation history to HISTORY_WINDOW messages."""
    if not history:
        return []
    cleaned = []
    for m in history[-HISTORY_WINDOW:]:
        cleaned.append({
            "role": m.get("role", "user"),
            "content": m.get("content", "")
        })
    return cleaned


# =========================
# TOP-N PARSER
# =========================

def _parse_top_n(question: str, default: int = 10) -> int:
    """
    Extracts the requested N from ranking questions.
    Handles: top 10, top-10, top_10, 10 countries, show me 5, give 20, etc.
    """
    q = question.lower()

    # Pattern 1: "top N" (with optional separator)
    m = re.search(r"\btop[\s\-_]?(\d+)\b", q)
    if m:
        return min(int(m.group(1)), 200)

    # Pattern 2: "N countries/regions/..." (number directly before entity word)
    m = re.search(
        r"\b(\d+)\s+(?:countries|regions|hottest|coolest|warmest|fastest|coldest|slowest)\b", q
    )
    if m:
        return min(int(m.group(1)), 200)

    # Pattern 3: verb + N (show me 5, list 10, give 20, find 15)
    m = re.search(
        r"\b(?:show|give|list|find|get|fetch|display)\s+(?:me\s+)?(\d+)\b", q
    )
    if m:
        return min(int(m.group(1)), 200)

    # Pattern 4: standalone number before or after ranking vocabulary
    m = re.search(
        r"\b(\d+)\b.*?\b(?:countries|regions|results|entries|rows|records)\b", q
    )
    if m:
        return min(int(m.group(1)), 200)

    return default


# =========================
# QUESTION TYPE CLASSIFIER
# =========================

def _classify_question_type(question: str) -> str:
    """
    Returns one of: 'stat', 'rank', 'compare', 'filter', 'trend',
                    'meta', 'summary', 'general'
    """
    q = question.lower()

    meta_kw = [
        "how many columns", "how many countries", "how many regions",
        "how many years", "what columns", "date range", "what is in",
        "describe the dataset", "dataset summary", "schema", "what data",
        "what fields", "list columns", "list countries", "list regions",
        "all countries", "all regions", "total countries", "total regions",
        "column names", "field names", "what are the columns",
        "which columns", "show columns", "how many records",
        "how many data points", "what years", "years covered",
        "year range", "what countries are", "countries in the dataset",
        "countries included", "regions in the dataset", "list all",
        "show all", "all the countries", "all the regions",
        "complete list", "full list", "show me the columns",
        "how many entries", "how many rows",
    ]
    if any(k in q for k in meta_kw):
        return "meta"

    summary_kw = [
        "summarize", "summary", "tell me about", "overview of",
        "profile of", "give me info", "what do you know about",
        "information about", "insights about", "analysis of",
        "describe", "explain", "break down",
    ]
    if any(k in q for k in summary_kw):
        return "summary"

    stat_kw = [
        "mean", "median", "average", "std", "standard deviation",
        "variance", "minimum", "maximum", "range", "percentile",
        "distribution", "statistics", "stats", "deviation",
    ]
    if any(k in q for k in stat_kw):
        return "stat"

    rank_kw = [
        "top", "bottom", "rank", "ranking", "hottest", "coldest",
        "warmest", "coolest", "fastest", "slowest", "highest", "lowest",
        "most", "least",
    ]
    if any(k in q for k in rank_kw):
        return "rank"

    compare_kw = [
        "compare", "vs", "versus", "against", "difference between",
        "better than", "worse than", "both",
    ]
    if any(k in q for k in compare_kw):
        return "compare"

    trend_kw = [
        "trend", "over time", "over years", "change", "warming",
        "cooling", "increase", "decrease", "grow", "rise", "fall",
        "historical", "history", "since", "from", "to",
    ]
    if any(k in q for k in trend_kw):
        return "trend"

    filter_kw = [
        "in europe", "in asia", "in africa", "in america", "in oceania",
        "in the region", "countries in", "regions in", "which countries",
        "which regions", "show me countries", "show me regions",
    ]
    if any(k in q for k in filter_kw):
        return "filter"

    return "general"


# =========================
# LLM CALL  (FIX-6: corrected history injection)
# =========================

def _llm_answer(system_prompt: str, question: str = "", history: list = None) -> str:
    """
    Invoke the LLM with typed exception handling.
    FIX-6: history is now injected correctly — role determines message class,
            content comes from the history record, not the current question.
    """
    try:
        llm = get_llm()
        messages = [SystemMessage(content=system_prompt)]

        if history:
            for msg in history[-HISTORY_WINDOW:]:
                role = str(msg.get("role", "user")).lower()
                content = str(msg.get("content", "")).strip()
                if not content:
                    continue
                if role in ["assistant", "ai"]:
                    messages.append(AIMessage(content=content))
                else:
                    # FIX-6: was wrongly using `question` as content; use history content
                    messages.append(HumanMessage(content=content))

        if question:
            messages.append(HumanMessage(content=question))

        response = llm.invoke(messages)

        content = getattr(response, "content", None)
        if not content:
            logger.warning("LLM returned empty content for question: %.200s", question)
            return "AI temporarily unavailable. Climate data is loaded, so please try again in a few seconds."

        return content.strip()

    except (OpenAIConfigError, GeminiConfigError) as e:
        logger.error("LLM configuration error: %s", e)
        return "AI temporarily unavailable. Configuration error — please check the API key."

    except Exception as e:
        err_str = str(type(e).__name__)

        if openai is not None:
            if isinstance(e, openai.RateLimitError):
                logger.warning("LLM rate limited: %s", e)
                return "AI temporarily unavailable. Rate limit reached — please wait a moment and try again."
            if isinstance(e, openai.AuthenticationError):
                logger.error("LLM authentication failed — check OPENAI_API_KEY: %s", e)
                return "AI temporarily unavailable. Authentication error — please check your API key configuration."
            if isinstance(e, openai.APITimeoutError):
                logger.warning("LLM request timed out: %s", e)
                return "AI temporarily unavailable. The request timed out — please try again in a few seconds."
            if isinstance(e, openai.BadRequestError):
                logger.error("LLM bad request (possibly context overflow): %s", e)
                return "AI temporarily unavailable. The request was too large — please ask a more specific question."

        logger.exception("LLM failed with unexpected error (%s): %s", err_str, e)
        return "AI temporarily unavailable. Climate data is loaded, so please try again in a few seconds."


# =========================
# ROUTER  (legacy — kept for compatibility)
# =========================

def route_query(question: str):
    q = question.lower()
    weather_keywords = ["weather", "temperature", "rain", "forecast", "humidity"]
    if any(k in q for k in weather_keywords):
        return "WEATHER"
    if any(k in q for k in ["why", "how", "explain", "reason"]):
        return "FOLLOWUP"
    return "CLIMATE"


def handle_weather(question, default_location=None):
    location = extract_location(question) or default_location
    if not location:
        return "Please provide a city or country name."
    weather = get_current_weather(location)
    return _format_weather_answer(weather)


def handle_climate(question, data):
    evidence = _build_climate_evidence(
        question,
        data.get("main"),
        data.get("country_trends"),
        data.get("regional_trends"),
    )
    if not evidence["text"]:
        return "No relevant climate data found."

    system_prompt = f"""You are EcoLens Climate AI. Use ONLY this dataset:

{evidence["text"]}

Answer the user's question based only on the data above. Be detailed and structured.
End with: "\n\nSource: Historical Climate Dataset"
"""
    return _llm_answer(system_prompt, question)


def handle_followup(question, history, last_state, data):
    last_answer = history[-1]["content"] if history else ""
    system_prompt = f"""You are EcoLens Climate AI continuing a previous answer.

PREVIOUS ANSWER:
{last_answer}

DATA:
{data}

RULES:
- Use previous answer context.
- Do NOT restart from zero.
- Explain the reasoning behind the previous answer in detail.
"""
    return _llm_answer(system_prompt, question, history)


# =========================
# MAIN ENTRY
# =========================

def ask_ecolens(question, history=None, data=None, analysis_state=None, default_location=None):

    history = history or []
    data = data or {}

    if not isinstance(analysis_state, dict):
        logger.warning(
            "analysis_state was not a dict (type=%s) — resetting to empty.",
            type(analysis_state).__name__,
        )
        analysis_state = {}
    else:
        analysis_state = analysis_state or {}

    try:
        if not data:
            from app.data_loader import clear_dataset_cache
            clear_dataset_cache()
            data = load_all_datasets(force_reload=True, retries=3, retry_delay=1)
    except Exception as exc:
        logger.warning("Climate datasets not ready for AI: %s", exc)
        return {
            "response": "Data is still loading. Please try again in a few seconds.",
            "analysis_state": analysis_state,
            "source": CLIMATE_SOURCE,
            "success": False,
        }

    answer_mode = _select_answer_mode(question, history, analysis_state, data)

    if answer_mode == "out_of_scope":
        return {
            "response": "I can only answer climate and weather related questions.",
            "analysis_state": analysis_state,
            "source": "EcoLens",
            "data_source": "EcoLens",
            "success": False,
        }
    if answer_mode == "historical_weather":
        return _answer_historical_weather(
            question=question,
            history=history,
            analysis_state=analysis_state,
        ) 

    if answer_mode == "weather":
        return _answer_weather_only(
            question=question,
            history=history,
            analysis_state=analysis_state,
            default_location=default_location,
        )

    if answer_mode == "climate":
        return _answer_climate_only(
            question=question,
            data=data,
            history=history,
            analysis_state=analysis_state,
        )

    if answer_mode == "hybrid":
        return _answer_weather_and_climate(
            question=question,
            default_location=default_location,
            data=data,
            history=history,
            analysis_state=analysis_state,
        )

    logger.error(
        "ask_ecolens reached unhandled answer_mode=%r for question: %.200s",
        answer_mode, question,
    )
    return {
        "response": (
            "I was unable to process your request. "
            "Please try rephrasing your question about climate or weather."
        ),
        "analysis_state": analysis_state,
        "source": "EcoLens",
        "data_source": "EcoLens",
        "success": False,
    }


# =========================
# DATAFRAME UTILITIES
# =========================

def safe_preview(df, n=20):
    """Module-level preview helper."""
    if df is None:
        return "No data available"
    try:
        return df.head(n).to_string(index=False)
    except Exception as e:
        return str(e)


def filter_relevant_data(
    question: str,
    df: Optional[pd.DataFrame] = None,
    max_rows: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """Filter df rows relevant to the question."""
    if df is None or not isinstance(df, pd.DataFrame):
        return None

    try:
        q = question.lower()
        top_n = _parse_top_n(question, default=10)
        limit = max_rows if max_rows is not None else max(top_n, 20)

        ranking_keywords = [
            "top", "highest", "lowest", "hottest", "coldest",
            "rank", "ranking", "trend", "compare",
        ]
        if any(word in q for word in ranking_keywords):
            return df  # caller will apply top_n limit

        words = re.findall(r"[a-zA-Z]+", q)
        stop_words = {
            "what", "is", "the", "a", "an", "of", "in", "on",
            "and", "to", "for", "why", "how", "tell", "me",
        }
        words = [w for w in words if w not in stop_words]
        if not words:
            return df

        def match_row(row):
            text = " ".join(map(str, row.values)).lower()
            return any(w in text for w in words)

        filtered = df[df.apply(match_row, axis=1)]
        if filtered.empty:
            return df
        return filtered.head(limit)

    except Exception:
        return df


# ==================================================
# FOLLOW-UP DETECTION
# ==================================================

FOLLOWUP_WORDS = [
    "why", "reason", "what about", "tell me more", "elaborate",
    "continue", "details", "more info",
    "this region", "that region", "this country", "that country",
    "why is that", "what does that mean", "explain that", "explain it",
]


def _is_followup_question(question: str) -> bool:
    """
    FIX-5: More conservative follow-up detection.
    Only treat as follow-up if there's a clear pronoun/reference with NO new entity.
    """
    q = str(question).lower()

    # If question contains a year, location, or country name → fresh question
    if any(word in q for word in ["this year", "latest year", "last year", "current year"]):
        return False
    if extract_location(question) or re.search(r"\b(19\d{2}|20\d{2})\b", q):
        return False
    if re.search(r"\b(last|past|previous)\s+\d+\s+years?\b", q):
        return True

    # If question contains clear ranking/meta/comparison keywords → fresh question
    fresh_signals = [
        "top ", "how many", "list all", "all countries", "all regions",
        "compare", "vs ", "versus", "rank", "ranking", "hottest", "coldest",
        "warmest", "coolest", "highest", "lowest", "fastest", "slowest",
    ]
    if any(signal in q for signal in fresh_signals):
        return False

    return any(word in q for word in FOLLOWUP_WORDS)


def _select_answer_mode(question: str, history=None, analysis_state=None, data=None) -> str:
    q = str(question).lower()
    # TEMP DEBUG — remove after testing
    _hist_date = extract_historical_date(question)
    _location = extract_location(question)
    print(f"DEBUG >>> hist_date={_hist_date!r}  location={_location!r}  q={question!r}") 
    analysis_state = analysis_state or {}
    data = data or {}
    has_year = bool(re.search(r"\b(19\d{2}|20\d{2})\b", q))

    csv_keywords = [
        "csv", "dataset", "data", "record", "records", "column", "columns",
        "summary", "summarize", "country", "countries", "climate", "warming",
        "cooling", "coolest", "coldest", "trend", "trends", "historical",
        "history", "average", "median", "maximum", "minimum", "standard deviation",
        "deviation", "range", "outlier", "outliers", "rank", "ranking", "highest",
        "lowest", "increase", "decrease", "stable", "stability", "affected",
        "findings", "insights", "patterns", "volatility", "hottest", "fastest",
        "region", "regions", "country climate", "how many", "top",
    ]

    live_weather_keywords = [
        "weather", "today", "now", "live", "current", "forecast", "rain",
        "humidity", "wind", "condition", "conditions", "feels like",
    ]

    temperature_keywords = ["temperature", "temp", "hot", "cold"]

    hybrid_keywords = [
        "weather and climate",
        "live and csv", "weather with csv",
    ]

    dataset_time_words = [
        "this year", "latest year", "latest", "last year", "past",
        "yearly", "over years", "historical", "history",
    ]

    # ── STEP 1: Historical weather — unconditionally first ──────────────
    # Do NOT rely solely on is_historical_weather_question() because it
    # requires both a date AND weather keywords — "temperature" alone
    # may not be in its keyword list depending on the query phrasing.
    # Instead, check directly: has a parseable historical date + a location.
    _hist_date = extract_historical_date(question)
    _location  = extract_location(question)

    if _hist_date and _location:
        # Confirm there's some weather/temperature intent
        _weather_intent_words = [
            "weather", "temperature", "temp", "hot", "cold", "rain",
            "humidity", "wind", "condition", "forecast", "climate",
            "warm", "cool", "rainfall", "storm",
        ]
        if any(w in q for w in _weather_intent_words):
            return "historical_weather"

    # ── STEP 2: Year present → climate, BUT only if not a weather+location Q ─
    # Without this guard, "temperature in Chennai in 1995" hits climate.
    _is_weather_and_location = (
        any(w in q for w in ["temperature", "temp", "weather", "hot", "cold",
                              "rain", "humidity", "wind", "condition", "forecast"])
        and bool(_location)
    )

    if not _is_weather_and_location:
        if has_year or any(word in q for word in dataset_time_words):
            return "climate"
    else:
        # Has year + weather + location but no parseable specific date
        # (e.g. "temperature in Chennai in 2022") → historical_weather
        if has_year:
            return "historical_weather"
        if any(word in q for word in dataset_time_words):
            return "climate"

    # ── STEP 3: Follow-up ───────────────────────────────────────────────
    if _is_followup_question(question):
        last_source = analysis_state.get("last_source")
        if last_source in {"weather", "climate"}:
            return last_source
        if last_source == "hybrid":
            return "climate"

    # ── STEP 4: Keyword routing ─────────────────────────────────────────
    wants_csv          = any(word in q for word in csv_keywords)
    wants_live_weather = any(word in q for word in live_weather_keywords)
    asks_temperature   = any(word in q for word in temperature_keywords)
    matches_dataset_place = _question_matches_dataset_place(question, data)

    wants_hybrid = any(word in q for word in hybrid_keywords) or (
        wants_live_weather
        and any(word in q for word in ["csv", "dataset", "climate trend", "historical"])
    )

    if wants_hybrid:
        return "hybrid"
    if wants_live_weather and not wants_csv:
        return "weather"
    if wants_csv or matches_dataset_place:
        return "climate"
    if wants_live_weather or (asks_temperature and _location):
        return "weather"

    climate_terms = [
        "climate", "temperature", "warming", "weather", "rain", "humidity",
        "forecast", "emission", "emissions", "region", "country", "trend",
        "trends", "dataset", "csv", "environment", "global warming",
    ]

    if any(term in q for term in climate_terms):
        return "climate"

    return "out_of_scope" 

def _question_matches_dataset_place(question: str, data=None) -> bool:
    data = data or {}
    q = str(question).lower()
    for df_name, column in [
        ("country_trends", "Country"),
        ("country_trends", "Region"),
        ("regional_trends", "Region"),
        ("main", "Country"),
        ("main", "Region"),
    ]:
        df = data.get(df_name)
        if df is None or getattr(df, "empty", True) or column not in df.columns:
            continue
        for value in df[column].dropna().astype(str).unique().tolist():
            value_text = value.lower().strip()
            if len(value_text) < 3:
                continue
            if re.search(rf"\b{re.escape(value_text)}\b", q):
                return True
            if _phrase_fuzzy_match(value_text, q):
                return True
    return False


def _normalize_match_text(value) -> str:
    return " ".join(re.sub(r"[^a-zA-Z0-9\s]", " ", str(value).lower()).split())


def _phrase_fuzzy_match(candidate: str, query: str, threshold: float = 0.88) -> bool:
    candidate = _normalize_match_text(candidate)
    query = _normalize_match_text(query)
    if not candidate or not query:
        return False

    candidate_words = candidate.split()
    n = len(candidate_words)

    if n == 1:
        return bool(re.search(rf"\b{re.escape(candidate)}\b", query))

    query_words = query.split()
    for i in range(len(query_words) - n + 1):
        phrase = " ".join(query_words[i: i + n])
        if phrase == candidate:
            return True
        if SequenceMatcher(None, candidate, phrase).ratio() >= threshold:
            return True

    return False


def _format_history(history, limit=8):
    lines = []
    for msg in (history or [])[-limit:]:
        role = str(msg.get("role", "user")).upper()
        content = str(msg.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) or "No previous conversation."


def _extract_country_or_region_terms(question: str):
    q = str(question).lower()
    words = re.sub(r"[^a-zA-Z0-9\s]", " ", q).split()
    stop_words = {
        "what", "is", "are", "the", "a", "an", "in", "of", "for", "to",
        "from", "using", "based", "on", "csv", "dataset", "data", "climate",
        "weather", "trend", "trends", "warming", "temperature", "average",
        "show", "tell", "me", "about", "compare", "with", "and", "today",
        "now", "live", "current", "how", "why", "explain", "region",
        "regions", "country", "countries", "year", "years", "change",
        "changed", "latest", "first", "last", "fastest", "hottest",
        "warmest", "top", "rank", "ranking",
        "emissions", "emission", "emitting", "emitted",
        "highest", "lowest", "most", "least", "best", "worst",
        "by", "per", "each", "all", "every", "any", "some",
        "coldest", "coolest", "warmest", "hottest", "slowest", "fastest",
        "give", "list", "show", "find", "get", "fetch",
    }
    return [word for word in words if word not in stop_words and len(word) > 2]


def _filter_rows_for_question(df, columns, question, max_rows=20):
    if df is None or getattr(df, "empty", True):
        return None
    available_columns = [col for col in columns if col in df.columns]
    if not available_columns:
        return df.head(max_rows)

    q = str(question).lower()

    # Phase 1: exact word-boundary entity match
    for col in available_columns:
        exact_mask = pd.Series(False, index=df.index)
        values = sorted(
            df[col].dropna().astype(str).unique().tolist(),
            key=len, reverse=True
        )
        for value in values:
            value_text = value.lower().strip()
            if len(value_text) < 3:
                continue
            if re.search(rf"\b{re.escape(value_text)}\b", q) or _phrase_fuzzy_match(value_text, q):
                exact_mask = exact_mask | (df[col].astype(str).str.lower() == value_text)
        exact_matches = df[exact_mask]
        if not exact_matches.empty:
            return exact_matches.head(max_rows)

    # Phase 2: domain-term-filtered keyword match
    terms = _extract_country_or_region_terms(question)
    if terms:
        mask = pd.Series(False, index=df.index)
        for col in available_columns:
            series = df[col].astype(str).str.lower()
            for term in terms:
                mask = mask | series.str.contains(re.escape(term), na=False)
        matched = df[mask]
        if not matched.empty:
            return matched.head(max_rows)

    # Phase 3: return full df for global/ranking questions
    return df.head(max_rows)


def _requested_csv_year(question: str, yearly_df=None):
    q = str(question).lower()
    explicit_year = re.search(r"\b(19\d{2}|20\d{2})\b", q)
    if explicit_year:
        return int(explicit_year.group())
    latest_words = ["this year", "latest year", "current year", "most recent year"]
    if any(word in q for word in latest_words):
        if yearly_df is not None and not getattr(yearly_df, "empty", True) and "Year" in yearly_df.columns:
            years = pd.to_numeric(yearly_df["Year"], errors="coerce").dropna()
            if not years.empty:
                return int(years.max())
    return None


def _filter_yearly_rows_for_question(yearly_df, question, max_rows=20):
    if yearly_df is None or getattr(yearly_df, "empty", True):
        return None

    df = yearly_df.copy()
    requested_year = _requested_csv_year(question, yearly_df)

    if requested_year and "Year" in df.columns:
        years = pd.to_numeric(df["Year"], errors="coerce")
        year_filtered = df[years == requested_year]
        if not year_filtered.empty:
            df = year_filtered

    q = str(question).lower()

    if any(w in q for w in ["highest", "hottest", "warmest", "most", "top", "maximum", "max",
                             "lowest", "coldest", "coolest", "least", "minimum", "min"]):
        if "AvgTemperature" in df.columns:
            df = df.copy()
            df["_temp_sort"] = pd.to_numeric(df["AvgTemperature"], errors="coerce")
            ascending = any(w in q for w in ["lowest", "coldest", "coolest", "least", "minimum", "min"])
            df = df.sort_values("_temp_sort", ascending=ascending).drop(columns=["_temp_sort"])
            return df.head(max_rows)

    result = _filter_rows_for_question(df, ["Country", "Region"], question, max_rows=max_rows)

    if result is None or result.empty:
        return df.head(max_rows)

    return result


def _question_targets_region(question: str, regional_df=None) -> bool:
    q = str(question).lower()
    if "region" in q or "regional" in q:
        return True
    if regional_df is None or getattr(regional_df, "empty", True) or "Region" not in regional_df.columns:
        return False
    region_rows = _filter_rows_for_question(regional_df, ["Region"], question, max_rows=1)
    return region_rows is not None and not region_rows.empty


def _countries_for_regions(country_df, region_rows, max_rows=10):
    if (
        country_df is None
        or getattr(country_df, "empty", True)
        or region_rows is None
        or getattr(region_rows, "empty", True)
        or "Region" not in country_df.columns
        or "Region" not in region_rows.columns
    ):
        return country_df.head(0) if country_df is not None else None
    regions = region_rows["Region"].dropna().astype(str).unique().tolist()
    if not regions:
        return country_df.head(0)
    region_country_rows = country_df[country_df["Region"].astype(str).isin(regions)]
    if "TemperatureChange" in region_country_rows.columns:
        region_country_rows = region_country_rows.sort_values("TemperatureChange", ascending=False)
    return region_country_rows.head(max_rows)


def _regions_from_country_rows(country_rows):
    if country_rows is None or getattr(country_rows, "empty", True) or "Region" not in country_rows.columns:
        return []
    return country_rows["Region"].dropna().astype(str).unique().tolist()


def _filter_regions_for_countries(regional_df, country_rows, max_rows=10):
    if regional_df is None or getattr(regional_df, "empty", True):
        return None
    regions = _regions_from_country_rows(country_rows)
    if not regions or "Region" not in regional_df.columns:
        return regional_df.head(0)
    mask = regional_df["Region"].astype(str).isin(regions)
    return regional_df[mask].head(max_rows)


def _df_preview(df, max_rows=20):
    if df is None or getattr(df, "empty", True):
        return "No matching rows."
    try:
        return df.head(max_rows).to_string(index=False)
    except Exception:
        return "Dataset preview unavailable."


# =========================
# FIX-4: RANKING BUILDER — dedicated function to build top-N tables
# =========================

def _build_ranking_block(country_df, region_df, top_n: int, question: str) -> str:
    """
    Build a clean, complete ranking block for rank-type questions.
    Always returns ALL top_n rows — never truncated.
    FIX-4: this is called explicitly for rank questions so the LLM gets
            a tight, focused context rather than a large noisy block.
    """
    q = question.lower()
    lines = []

    asc_kw = ["coolest", "coldest", "lowest", "slowest", "least", "bottom"]
    desc_kw = ["hottest", "warmest", "highest", "fastest", "most", "top"]
    warming_kw = ["warming", "warm", "fastest", "change", "trend"]
    temp_kw = ["temperature", "hot", "cold", "warm", "cool"]
    region_kw = ["region", "regional", "continent"]

    wants_asc = any(k in q for k in asc_kw)
    wants_desc = any(k in q for k in desc_kw)
    wants_warming = any(k in q for k in warming_kw)
    wants_region = any(k in q for k in region_kw)

    # --- Country rankings ---
    if country_df is not None and not getattr(country_df, "empty", True) and not wants_region:
        if wants_warming:
            label = f"Top {top_n} fastest warming countries"
            col = "TemperatureChange"
            asc = wants_asc and not wants_desc
        else:
            label = f"Top {top_n} {'coolest/coldest' if wants_asc else 'hottest/warmest'} countries by average temperature"
            col = "AverageTemperature"
            asc = wants_asc

        ranked = _sort_df(country_df, col, ascending=asc)
        if ranked is not None and not getattr(ranked, "empty", True):
            lines.append(f"=== {label.upper()} ===")
            cols = [c for c in ["Country", "Region", col, "WarmingTrendPerYear", "AverageTemperature"]
                    if c in ranked.columns]
            lines.append(ranked.head(top_n)[cols].to_string(index=False))
            lines.append(f"\n(Showing {min(top_n, len(ranked))} of {len(ranked)} countries)")

        # Also include opposite direction for context
        if not wants_asc and not wants_desc:
            # neutral — show both
            lines.append(f"\n=== TOP {top_n} COOLEST COUNTRIES ===")
            opp = _sort_df(country_df, "AverageTemperature", ascending=True)
            if opp is not None and not getattr(opp, "empty", True):
                cols2 = [c for c in ["Country", "Region", "AverageTemperature"] if c in opp.columns]
                lines.append(opp.head(top_n)[cols2].to_string(index=False))

    # --- Region rankings ---
    if region_df is not None and not getattr(region_df, "empty", True):
        if wants_warming:
            label = f"Top {top_n} fastest warming regions"
            col = "RegionalTemperatureChange"
            asc = wants_asc and not wants_desc
        else:
            label = f"Top {top_n} {'coolest' if wants_asc else 'hottest'} regions by average temperature"
            col = "AverageRegionalTemperature"
            asc = wants_asc

        ranked_r = _sort_df(region_df, col, ascending=asc)
        if ranked_r is not None and not getattr(ranked_r, "empty", True):
            lines.append(f"\n=== {label.upper()} ===")
            cols_r = [c for c in ["Region", col, "RegionalWarmingTrendPerYear", "AverageRegionalTemperature"]
                      if c in ranked_r.columns]
            lines.append(ranked_r.head(top_n)[cols_r].to_string(index=False))

    return "\n".join(lines)


# =========================
# FIX-3: COMPARISON BUILDER — dedicated function for compare questions
# =========================

def _build_comparison_block(question: str, country_df, region_df) -> str:
    """
    Extract all entities mentioned in the question and build a side-by-side
    comparison table. Supports 2+ countries or 2+ regions.
    FIX-3: previously _matched_names_from_question was called but its output
            wasn't reliably wired into a comparison table the LLM could use.
    """
    if country_df is None and region_df is None:
        return ""

    q = str(question).lower()
    lines = []

    # Expand common aliases
    aliases = {
        "usa": "united states", "us ": "united states", "america": "united states",
        "uk": "united kingdom", "britain": "united kingdom",
        "uae": "united arab emirates",
    }
    expanded_q = q
    for alias, canonical in aliases.items():
        if alias in q:
            expanded_q += f" {canonical}"

    # Match country names
    matched_country_names = []
    if country_df is not None and not getattr(country_df, "empty", True) and "Country" in country_df.columns:
        for value in sorted(country_df["Country"].dropna().astype(str).unique(), key=len, reverse=True):
            norm = _normalize_match_text(value)
            if len(norm) < 3:
                continue
            if re.search(rf"\b{re.escape(norm)}\b", _normalize_match_text(expanded_q)):
                matched_country_names.append(value)

    # Match region names
    matched_region_names = []
    if region_df is not None and not getattr(region_df, "empty", True) and "Region" in region_df.columns:
        for value in sorted(region_df["Region"].dropna().astype(str).unique(), key=len, reverse=True):
            norm = _normalize_match_text(value)
            if len(norm) < 3:
                continue
            if re.search(rf"\b{re.escape(norm)}\b", _normalize_match_text(expanded_q)):
                matched_region_names.append(value)

    # Build country comparison table
    if len(matched_country_names) >= 2 and country_df is not None:
        rows = country_df[country_df["Country"].astype(str).isin(matched_country_names)]
        if not rows.empty:
            lines.append("=== COUNTRY COMPARISON TABLE ===")
            compare_cols = [c for c in [
                "Country", "Region", "FirstYear", "LatestYear",
                "FirstTemperature", "LatestTemperature", "AverageTemperature",
                "TemperatureChange", "WarmingTrendPerYear", "VolatilityScore",
            ] if c in rows.columns]
            lines.append(rows[compare_cols].to_string(index=False))

            # Numerical diff lines
            lines.append("\n=== DIRECT COMPARISON (calculated differences) ===")
            records = rows.set_index("Country").to_dict(orient="index")
            for i, a_name in enumerate(matched_country_names):
                for b_name in matched_country_names[i + 1:]:
                    a = records.get(a_name, {})
                    b = records.get(b_name, {})
                    if not a or not b:
                        continue
                    diffs = []
                    for col, label in [
                        ("AverageTemperature", "avg temp diff"),
                        ("LatestTemperature", "latest temp diff"),
                        ("TemperatureChange", "total warming diff"),
                        ("WarmingTrendPerYear", "warming trend diff"),
                    ]:
                        try:
                            va = float(a.get(col, 0) or 0)
                            vb = float(b.get(col, 0) or 0)
                            diffs.append(f"{label}: {va - vb:+.3f} C ({a_name} vs {b_name})")
                        except (TypeError, ValueError):
                            pass
                    if diffs:
                        lines.append(f"{a_name} vs {b_name}:")
                        lines.extend([f"  {d}" for d in diffs])

    # Build region comparison table
    if len(matched_region_names) >= 2 and region_df is not None:
        r_rows = region_df[region_df["Region"].astype(str).isin(matched_region_names)]
        if not r_rows.empty:
            lines.append("\n=== REGION COMPARISON TABLE ===")
            r_cols = [c for c in [
                "Region", "FirstYear", "LatestYear",
                "FirstRegionalTemperature", "LatestRegionalTemperature",
                "AverageRegionalTemperature", "RegionalTemperatureChange",
                "RegionalWarmingTrendPerYear", "RegionalVolatilityScore",
            ] if c in r_rows.columns]
            lines.append(r_rows[r_cols].to_string(index=False))

    if not lines:
        # Single entity or unrecognised — return empty so caller falls back
        return ""

    return "\n".join(lines)


def _build_dataset_facts(
    question,
    raw_df=None,
    yearly_df=None,
    country_df=None,
    region_df=None,
    top_n: int = 10,
) -> str:
    lines = []
    q = str(question).lower()
    raw_df = raw_df if raw_df is not None else pd.DataFrame()
    yearly_df = yearly_df if yearly_df is not None else pd.DataFrame()
    country_df = country_df if country_df is not None else pd.DataFrame()
    region_df = region_df if region_df is not None else pd.DataFrame()

    main = yearly_df if not getattr(yearly_df, "empty", True) else raw_df

    # ── YEAR-SPECIFIC BLOCK ──────────────────────────────────────────────
    requested_year = _requested_csv_year(question, main if not getattr(main, "empty", True) else None)
    if requested_year and main is not None and not getattr(main, "empty", True):
        lines.append(f"=== YEAR-SPECIFIC DATA: {requested_year} ===")
        year_df = main.copy()
        if "Year" in year_df.columns:
            year_df = year_df[pd.to_numeric(year_df["Year"], errors="coerce") == requested_year]

        if not year_df.empty:
            lines.append(f"Records for {requested_year}: {len(year_df)}")

            if "AvgTemperature" in year_df.columns and "Country" in year_df.columns:
                year_df = year_df.copy()
                year_df["_t"] = pd.to_numeric(year_df["AvgTemperature"], errors="coerce")
                sorted_hot = year_df.dropna(subset=["_t"]).sort_values("_t", ascending=False)
                sorted_cold = year_df.dropna(subset=["_t"]).sort_values("_t", ascending=True)

                lines.append(f"\nTop {top_n} highest temperature countries in {requested_year}:")
                lines.append(_records_table(sorted_hot, ["Country", "Region", "AvgTemperature"], top_n))
                lines.append(f"\nTop {top_n} lowest temperature countries in {requested_year}:")
                lines.append(_records_table(sorted_cold, ["Country", "Region", "AvgTemperature"], top_n))

                if not sorted_hot.empty:
                    top = sorted_hot.iloc[0]
                    lines.append(
                        f"\nHighest temperature country in {requested_year}: "
                        f"{top.get('Country', 'N/A')} with {float(top['_t']):.3f} C"
                    )
                if not sorted_cold.empty:
                    bot = sorted_cold.iloc[0]
                    lines.append(
                        f"Lowest temperature country in {requested_year}: "
                        f"{bot.get('Country', 'N/A')} with {float(bot['_t']):.3f} C"
                    )

                temp_vals = year_df["_t"].dropna()
                if not temp_vals.empty:
                    lines.append(
                        f"Global average temperature in {requested_year}: {temp_vals.mean():.3f} C"
                    )
        else:
            lines.append(f"No records found for year {requested_year} in the dataset.")

    # ── METADATA BLOCK ──────────────────────────────────────────────────
    if main is not None and not getattr(main, "empty", True):
        lines.append("\n=== DATASET METADATA ===")
        lines.append(f"Total records: {len(main)}")
        lines.append(f"Columns: {', '.join(map(str, main.columns.tolist()))}")

        if "Country" in main.columns:
            countries = sorted(main["Country"].dropna().astype(str).unique().tolist())
            lines.append(f"Number of countries: {len(countries)}")
            lines.append(f"All countries: {', '.join(countries)}")

        if "Region" in main.columns:
            regions = sorted(main["Region"].dropna().astype(str).unique().tolist())
            lines.append(f"Number of regions: {len(regions)}")
            lines.append(f"All regions: {', '.join(regions)}")

        if "Year" in main.columns:
            years = pd.to_numeric(main["Year"], errors="coerce").dropna()
            if not years.empty:
                lines.append(f"Date range: {int(years.min())} to {int(years.max())}")

        if "AvgTemperature" in main.columns:
            temp = pd.to_numeric(main["AvgTemperature"], errors="coerce").dropna()
            if not temp.empty:
                lines.append(
                    f"Temperature stats: mean {temp.mean():.3f} C, median {temp.median():.3f} C, "
                    f"min {temp.min():.3f} C, max {temp.max():.3f} C, "
                    f"std {temp.std():.3f} C, range {(temp.max() - temp.min()):.3f} C"
                )

        if {"Country", "Year", "AvgTemperature"}.issubset(main.columns):
            latest_year = int(pd.to_numeric(main["Year"], errors="coerce").max())
            latest_rows = main[pd.to_numeric(main["Year"], errors="coerce") == latest_year]
            if not latest_rows.empty:
                hottest_latest = latest_rows.loc[
                    pd.to_numeric(latest_rows["AvgTemperature"], errors="coerce").idxmax()
                ]
                lines.append(
                    f"Highest country temperature in latest year ({latest_year}): "
                    f"{hottest_latest['Country']} at {float(hottest_latest['AvgTemperature']):.3f} C"
                )

            yearly_summary = (
                main.assign(AvgTemperature=pd.to_numeric(main["AvgTemperature"], errors="coerce"))
                .groupby("Year", as_index=False)["AvgTemperature"]
                .mean()
                .dropna()
                .sort_values("Year")
            )
            if not yearly_summary.empty:
                high = yearly_summary.loc[yearly_summary["AvgTemperature"].idxmax()]
                low = yearly_summary.loc[yearly_summary["AvgTemperature"].idxmin()]
                first = yearly_summary.iloc[0]
                latest = yearly_summary.iloc[-1]
                lines.append(
                    f"Overall yearly trend: {int(first['Year'])} avg {first['AvgTemperature']:.3f} C → "
                    f"{int(latest['Year'])} avg {latest['AvgTemperature']:.3f} C "
                    f"(change {(latest['AvgTemperature'] - first['AvgTemperature']):.3f} C)"
                )
                lines.append(
                    f"Highest global-average year: {int(high['Year'])} at {high['AvgTemperature']:.3f} C; "
                    f"lowest: {int(low['Year'])} at {low['AvgTemperature']:.3f} C"
                )
                lines.append("Yearly average temperatures (last 12 years):")
                lines.append(_records_table(yearly_summary.tail(12), ["Year", "AvgTemperature"], max_rows=12))

    # ── COUNTRY RANKINGS ────────────────────────────────────────────────
    if country_df is not None and not getattr(country_df, "empty", True):
        lines.append(f"\n=== COUNTRY RANKINGS (top {top_n}) ===")

        lines.append(f"Top {top_n} hottest countries (by average temperature):")
        lines.append(_records_table(
            _sort_df(country_df, "AverageTemperature", False),
            ["Country", "Region", "AverageTemperature"], top_n,
        ))

        lines.append(f"\nTop {top_n} coolest countries (by average temperature):")
        lines.append(_records_table(
            _sort_df(country_df, "AverageTemperature", True),
            ["Country", "Region", "AverageTemperature"], top_n,
        ))

        lines.append(f"\nTop {top_n} fastest warming countries:")
        lines.append(_records_table(
            _sort_df(country_df, "TemperatureChange", False),
            ["Country", "Region", "TemperatureChange", "WarmingTrendPerYear"], top_n,
        ))

        cooling = (
            country_df[pd.to_numeric(country_df.get("TemperatureChange"), errors="coerce") < 0]
            if "TemperatureChange" in country_df.columns
            else country_df.head(0)
        )
        lines.append(f"\nTop {top_n} cooling / negative-change countries:")
        lines.append(_records_table(
            _sort_df(cooling, "TemperatureChange", True),
            ["Country", "Region", "TemperatureChange", "WarmingTrendPerYear"], top_n,
        ))

        lines.append(f"\nTop {top_n} most stable countries (lowest volatility):")
        lines.append(_records_table(
            _sort_df(country_df, "VolatilityScore", True),
            ["Country", "Region", "VolatilityScore", "TemperatureChange"], top_n,
        ))

        matched_countries = _matched_names_from_question(q, country_df, "Country")
        if matched_countries:
            matched_rows = country_df[country_df["Country"].astype(str).isin(matched_countries)]
            lines.append("\nMatched country details:")
            lines.append(_records_table(matched_rows, [
                "Country", "Region", "FirstYear", "LatestYear", "FirstTemperature",
                "LatestTemperature", "AverageTemperature", "TemperatureChange",
                "WarmingTrendPerYear", "VolatilityScore",
            ], max(8, len(matched_countries))))

        if len(matched_countries) >= 2:
            lines.extend(_country_difference_lines(country_df, matched_countries[:4]))

    # ── REGIONAL RANKINGS ───────────────────────────────────────────────
    if region_df is not None and not getattr(region_df, "empty", True):
        lines.append(f"\n=== REGIONAL RANKINGS (top {top_n}) ===")

        lines.append("Average temperature by region (hottest first):")
        lines.append(_records_table(
            _sort_df(region_df, "AverageRegionalTemperature", False),
            ["Region", "AverageRegionalTemperature", "RegionalTemperatureChange", "RegionalWarmingTrendPerYear"],
            top_n,
        ))

        lines.append(f"\nTop {top_n} fastest warming regions:")
        lines.append(_records_table(
            _sort_df(region_df, "RegionalTemperatureChange", False),
            ["Region", "RegionalTemperatureChange", "RegionalWarmingTrendPerYear"],
            top_n,
        ))

        matched_regions = _matched_names_from_question(q, region_df, "Region")
        if matched_regions:
            matched_rows = region_df[region_df["Region"].astype(str).isin(matched_regions)]
            lines.append("\nMatched region details:")
            lines.append(_records_table(matched_rows, [
                "Region", "FirstYear", "LatestYear", "FirstRegionalTemperature",
                "LatestRegionalTemperature", "AverageRegionalTemperature",
                "RegionalTemperatureChange", "RegionalWarmingTrendPerYear", "RegionalVolatilityScore",
            ], max(8, len(matched_regions))))

    return "\n".join(line for line in lines if line)


def _records_table(df, columns, max_rows=10):
    if df is None or getattr(df, "empty", True):
        return "No matching rows."
    available = [col for col in columns if col in df.columns]
    if not available:
        return "No matching columns."
    try:
        return df.head(max_rows)[available].to_string(index=False)
    except Exception:
        return "Table unavailable."


def _sort_df(df, column, ascending):
    if df is None or getattr(df, "empty", True) or column not in df.columns:
        return df
    sorted_df = df.copy()
    sorted_df["__sort__"] = pd.to_numeric(sorted_df[column], errors="coerce")
    result = (
        sorted_df
        .dropna(subset=["__sort__"])
        .sort_values("__sort__", ascending=ascending)
        .drop(columns=["__sort__"])
    )
    return result


def _matched_names_from_question(q, df, column):
    if df is None or getattr(df, "empty", True) or column not in df.columns:
        return []
    matches = []
    aliases = {
        "usa": "united states",
        "us": "united states",
        "u s": "united states",
        "america": "united states",
        "uk": "united kingdom",
        "britain": "united kingdom",
    }
    normalized_q = _normalize_match_text(q)
    expanded_q = normalized_q
    for alias, canonical in aliases.items():
        if re.search(rf"\b{re.escape(alias)}\b", normalized_q):
            expanded_q += f" {canonical}"
    for value in df[column].dropna().astype(str).unique().tolist():
        normalized_value = _normalize_match_text(value)
        if re.search(rf"\b{re.escape(normalized_value)}\b", expanded_q) or _phrase_fuzzy_match(normalized_value, expanded_q):
            matches.append(value)
    return matches


def _country_difference_lines(country_df, countries):
    lines = []
    rows = country_df[country_df["Country"].astype(str).isin(countries)]
    records = rows.set_index("Country").to_dict(orient="index")
    for i, first in enumerate(countries):
        for second in countries[i + 1:]:
            a = records.get(first)
            b = records.get(second)
            if not a or not b:
                continue
            avg_diff = float(a.get("AverageTemperature", 0)) - float(b.get("AverageTemperature", 0))
            latest_diff = float(a.get("LatestTemperature", 0)) - float(b.get("LatestTemperature", 0))
            change_diff = float(a.get("TemperatureChange", 0)) - float(b.get("TemperatureChange", 0))
            lines.append(
                f"Comparison {first} vs {second}: avg diff {avg_diff:.3f} C, "
                f"latest-year diff {latest_diff:.3f} C, warming-change diff {change_diff:.3f} C."
            )
    return lines


def _plain_weather_context(weather: dict) -> str:
    if not weather:
        return "Weather data unavailable."
    normalized = normalize_weather(weather) or {}
    place = normalized.get("location", {})
    current = normalized.get("current", {})
    today = normalized.get("today", {})
    label = ", ".join(
        part for part in [place.get("name"), place.get("admin1"), place.get("country")] if part
    ) or "Unknown"
    return (
        f"Location: {label}\n"
        f"Date: {weather.get('date', 'Unknown')}\n"
        f"Temperature: {current.get('temperature_c', 'N/A')} C\n"
        f"Feels Like: {current.get('feels_like_c', 'N/A')} C\n"
        f"Condition: {current.get('condition', 'Unknown')}\n"
        f"Humidity: {current.get('humidity_percent', 'N/A')}%\n"
        f"Wind Speed: {current.get('wind_speed_kmh', 'N/A')} km/h\n"
        f"Today Min: {today.get('min_temperature_c', 'N/A')} C\n"
        f"Today Max: {today.get('max_temperature_c', 'N/A')} C\n"
        f"Rain: {today.get('rain_sum_mm', 'N/A')} mm"
    )


def _state_for_answer(
    question,
    answer,
    source,
    location=None,
    weather=None,
    followup=False,
    chain_name=None,
    evidence=None,
):
    return {
        "last_question": question,
        "last_answer": answer[:MAX_LAST_ANSWER_CHARS],
        "followup": followup,
        "last_source": source,
        "last_location": location,
        "last_weather": weather or {},
        "last_chain": chain_name,
        "last_evidence": evidence[:MAX_LAST_EVIDENCE_CHARS] if isinstance(evidence, str) else "",
    }


def _answer_weather_only(question, history=None, analysis_state=None, default_location=None):
    analysis_state = analysis_state or {}
    is_followup = _is_followup_question(question)
    location = (
        extract_location(question)
        or analysis_state.get("last_location")
        or default_location
    )

    if not location:
        answer = "Please include a city name so I can fetch the live weather.\n\nSource: Live Weather"
        return {
            "response": answer,
            "analysis_state": _state_for_answer(question, answer, "weather", followup=is_followup),
            "source": WEATHER_SOURCE,
            "data_source": WEATHER_SOURCE,
            "success": False,
        }

    try:
        weather = get_current_weather(location)
    except Exception as exc:
        logger.warning("Weather unavailable for AI query: %s", exc)
        answer = "Live weather is temporarily unavailable. Please try again in a few seconds.\n\nSource: Live Weather"
        return {
            "response": answer,
            "analysis_state": _state_for_answer(question, answer, "weather", location=location, followup=is_followup),
            "source": WEATHER_SOURCE,
            "data_source": WEATHER_SOURCE,
            "success": False,
        }

    weather_context = _plain_weather_context(weather)

    # FIX-1: only inject last_answer for genuine follow-ups
    prior_section = ""
    if is_followup and analysis_state.get("last_answer"):
        prior_section = f"\nPREVIOUS ANSWER (for follow-up continuity):\n{analysis_state['last_answer'][:MAX_LAST_ANSWER_CHARS]}\n"

    system_prompt = f"""You are EcoLens AI. Answer like a helpful conversational assistant.

Use ONLY the live weather data below. Do not add CSV, climate trend, or historical dataset information.
{prior_section}
LIVE WEATHER:
{weather_context}

OUTPUT FORMAT RULES:
- Start with a ## heading: ## Weather - <location name> 
- Then write one natural sentence summary.
- Then output each field on its own line as: **Label**: value
  Use these exact labels: Temperature, Feels Like, Condition, Humidity, Wind Speed, Today Min, Today Max, Rain
- Add a ## Key Observations heading followed by 2-3 bullet points noting anything notable.
- Do NOT use markdown tables.
- End with: Source: Live Weather
"""
    answer = _llm_answer(system_prompt, question, history)

    if answer.startswith("AI temporarily unavailable"):
        answer = f"Using live weather data,\n\n{weather_context}\n\nSource: Live Weather"

    return {
        "response": answer,
        "formatted_response": _fmt_response(answer, WEATHER_SOURCE) if _FMT else answer,
        "analysis_state": _state_for_answer(
            question, answer, "weather", location=location, weather=weather, followup=is_followup,
        ),
        "source": WEATHER_SOURCE,
        "data_source": WEATHER_SOURCE,
        "success": True,
    }
def _answer_historical_weather(question, history=None, analysis_state=None):
    analysis_state = analysis_state or {}

    location = extract_location(question)
    date_str  = extract_historical_date(question)

    if not location or not date_str:
        answer = (
            "Please provide both a location and a time period. "
            "Examples: 'temperature in Chennai in 2022', "
            "'weather in Delhi in March 2023', "
            "'what was the temperature in Mumbai on 5 January 2020'."
            "\n\nSource: Historical Weather"
        )
        return {
            "response": answer,
            "analysis_state": _state_for_answer(question, answer, "weather"),
            "source": "Historical Weather",
            "success": False,
        }

    try:
        weather = get_historical_weather(location, date_str)
    except Exception as exc:
        logger.warning("Historical weather failed: %s", exc)
        answer = (
            f"Historical weather data for {location} ({date_str}) is unavailable. "
            f"{str(exc)}\n\nSource: Historical Weather"
        )
        return {
            "response": answer,
            "analysis_state": _state_for_answer(question, answer, "weather", location=location),
            "source": "Historical Weather",
            "success": False,
        }

    current  = weather.get("current", {})
    today_w  = weather.get("today", {})
    place    = weather.get("location", {})
    mode     = weather.get("mode", "day")
    label    = weather.get("label", date_str)
    num_days = weather.get("num_days", 1)
    monthly  = weather.get("monthly_summary", [])

    location_label = ", ".join(
        p for p in [place.get("name"), place.get("admin1"), place.get("country")] if p
    )

    # Build context string depending on query mode
    if mode == "day":
        period_desc = f"on {label}"
        weather_context = (
            f"Location: {location_label}\n"
            f"Date: {label}\n"
            f"Mean Temperature: {current.get('temperature_c')} C\n"
            f"Max Temperature: {today_w.get('max_temperature_c')} C\n"
            f"Min Temperature: {today_w.get('min_temperature_c')} C\n"
            f"Condition: {current.get('condition')}\n"
            f"Wind Speed: {current.get('wind_speed_kmh')} km/h\n"
            f"Rainfall: {today_w.get('rain_sum_mm')} mm"
        )
    elif mode == "month":
        period_desc = f"in {label}"
        weather_context = (
            f"Location: {location_label}\n"
            f"Period: {label} ({num_days} days)\n"
            f"Average Temperature: {current.get('temperature_c')} C\n"
            f"Highest Daily Max: {today_w.get('max_temperature_c')} C\n"
            f"Lowest Daily Min: {today_w.get('min_temperature_c')} C\n"
            f"Total Rainfall: {today_w.get('rain_sum_mm')} mm\n"
            f"Avg Wind Speed: {current.get('wind_speed_kmh')} km/h\n"
            f"Dominant Condition: {current.get('condition')}"
        )
    else:  # year
        monthly_lines = "\n".join(
            f"  {row['month']}: {row['avg_temp']} C" for row in monthly
        ) if monthly else "  (not available)"
        period_desc = f"in {label}"
        weather_context = (
            f"Location: {location_label}\n"
            f"Year: {label} ({num_days} days of data)\n"
            f"Annual Average Temperature: {current.get('temperature_c')} C\n"
            f"Hottest Day (max): {today_w.get('max_temperature_c')} C\n"
            f"Coldest Day (min): {today_w.get('min_temperature_c')} C\n"
            f"Total Annual Rainfall: {today_w.get('rain_sum_mm')} mm\n"
            f"Avg Wind Speed: {current.get('wind_speed_kmh')} km/h\n"
            f"Monthly Breakdown:\n{monthly_lines}"
        )

    period_label_for_heading = f"{location_label} — {label}"

    system_prompt = f"""You are EcoLens AI. Answer like a helpful weather assistant.

HISTORICAL WEATHER DATA for {location_label} {period_desc}:
{weather_context}

OUTPUT FORMAT RULES:
- Start with: ## Historical Weather — {period_label_for_heading}
- Write one natural sentence summary of the period.
- Output each field on its own line as: **Label**: value
- For year queries, include a monthly breakdown table: | Month | Avg Temp (C) |
- Add ## Key Notes with 2-3 bullet points about notable aspects of the weather.
- End with: Source: Historical Weather Archive (Open-Meteo)
"""

    answer = _llm_answer(system_prompt, question, history)

    if answer.startswith("AI temporarily unavailable"):
        answer = (
            f"Historical weather for {location_label} {period_desc}:\n\n"
            f"{weather_context}\n\n"
            f"Source: Historical Weather Archive"
        )

    return {
        "response": answer,
        "formatted_response": _fmt_response(answer, "Historical Weather Archive") if _FMT else answer,
        "analysis_state": _state_for_answer(question, answer, "weather", location=location),
        "source": "Historical Weather Archive",
        "data_source": "Historical Weather Archive",
        "success": True,
    } 

# =========================
# CLIMATE ONLY  (FIX-1, FIX-2, FIX-3, FIX-4, FIX-5)
# =========================

def _answer_climate_only(question, data, history=None, analysis_state=None):
    analysis_state = analysis_state or {}
    data = data or {}
    is_followup = _is_followup_question(question)

    top_n = _parse_top_n(question, default=10)
    q_type = _classify_question_type(question)

    # FIX-5: Only enrich question with prior context for genuine follow-ups
    enriched_question = question
    prior_context_section = ""
    if is_followup and analysis_state.get("last_question"):
        prior_context_section = (
            f"PREVIOUS QUESTION: {analysis_state.get('last_question')}\n"
            f"PREVIOUS ANSWER SUMMARY: {analysis_state.get('last_answer', '')[:MAX_PRIOR_CONTEXT_CHARS]}\n"
            f"FOLLOW-UP: {question}"
        )
        enriched_question = (
            f"Previous question: {analysis_state.get('last_question')}\n"
            f"Previous answer summary: {analysis_state.get('last_answer', '')[:400]}\n"
            f"Follow-up question: {question}"
        )

    # ── Reasoning chain ─────────────────────────────────────────────────
    try:
        intent = detect_intent(enriched_question)
        if intent == Intent.WEATHER:
            intent = Intent.CLIMATE
    except Exception:
        intent = Intent.CLIMATE

    try:
        chain = build_reasoning_chain(
            intent=intent,
            question=enriched_question,
            location=analysis_state.get("last_location"),
            yearly_df=data.get("main"),
            country_trends=data.get("country_trends"),
            regional_trends=data.get("regional_trends"),
        )
    except Exception as exc:
        logger.warning("Climate reasoning chain failed: %s", exc)
        chain = {
            "chain_name": "climate",
            "evidence": analysis_state.get("last_evidence", "") if is_followup else "",
            "structured_facts": {},
        }

    # ── FIX-4: Build dedicated ranking block for rank questions ─────────
    ranking_block = ""
    if q_type == "rank":
        ranking_block = _build_ranking_block(
            data.get("country_trends"),
            data.get("regional_trends"),
            top_n,
            question,
        )

    # ── FIX-3: Build dedicated comparison block for compare questions ───
    comparison_block = ""
    if q_type == "compare":
        comparison_block = _build_comparison_block(
            question,
            data.get("country_trends"),
            data.get("regional_trends"),
        )

    # ── FIX-2: For meta/debug questions, build a focused metadata-only block
    meta_block = ""
    if q_type == "meta":
        meta_block = _build_meta_answer_block(question, data.get("main"), data.get("country_trends"), data.get("regional_trends"))

    # ── FULL dataset facts ───────────────────────────────────────────────
    # For rank/compare/meta, skip the heavy full_facts to keep context lean
    if q_type in ("rank", "compare", "meta"):
        full_facts = ""
    else:
        full_facts = _build_dataset_facts(
            enriched_question,
            raw_df=data.get("raw"),
            yearly_df=data.get("main"),
            country_df=data.get("country_trends"),
            region_df=data.get("regional_trends"),
            top_n=top_n,
        )

    full_facts_trimmed = full_facts[:MAX_FACTS_CHARS]

    # ── Targeted supplement: entities matched in the question ───────────
    filter_max = max(top_n, 20)

    # For rank/compare/meta, skip supplement too
    supplement = ""
    if q_type not in ("rank", "compare", "meta"):
        matched_main = _filter_yearly_rows_for_question(data.get("main"), enriched_question)
        matched_countries = _filter_rows_for_question(
            data.get("country_trends"), ["Country", "Region"], enriched_question, max_rows=filter_max
        )
        matched_regions = _filter_rows_for_question(
            data.get("regional_trends"), ["Region"], enriched_question, max_rows=filter_max
        )

        supplement_parts = []
        if matched_main is not None and not matched_main.empty:
            supplement_parts.append(f"MATCHED YEARLY ROWS:\n{_df_preview(matched_main, filter_max)}")
        if matched_countries is not None and not matched_countries.empty:
            supplement_parts.append(f"MATCHED COUNTRY DETAILS:\n{_df_preview(matched_countries, filter_max)}")
        if matched_regions is not None and not matched_regions.empty:
            supplement_parts.append(f"MATCHED REGION DETAILS:\n{_df_preview(matched_regions, filter_max)}")

        if supplement_parts:
            supplement = (
                "\n\n=== TARGETED MATCH (entities mentioned in question) ===\n"
                + "\n\n".join(supplement_parts)
            )
    else:
        matched_main = None
        matched_countries = None
        matched_regions = None

    supplement_trimmed = supplement[:MAX_SUPPLEMENT_CHARS]

    # ── Dynamic top-N (only for general questions) ──────────────────────
    dynamic_section = ""
    if q_type not in ("rank", "compare", "meta"):
        try:
            ct = data.get("country_trends")
            if ct is not None and not ct.empty:
                top_countries = get_top_countries(ct, n=top_n)
                if top_countries is not None and not top_countries.empty:
                    dynamic_section = (
                        f"\nDYNAMIC TOP {top_n} COUNTRIES:\n"
                        f"{_df_preview(top_countries, top_n)}\n"
                    )
        except Exception as exc:
            logger.warning("get_top_countries failed: %s", exc)

    # ── Evidence from chain ───────────────────────────────────────────────
    # FIX-5: Only use prior evidence for genuine follow-ups
    if is_followup:
        evidence_trimmed = chain.get("evidence", "")[:MAX_EVIDENCE_CHARS]
    else:
        evidence_trimmed = chain.get("evidence", "")[:MAX_EVIDENCE_CHARS]

    # FIX-1 & FIX-5: Only inject prior context for follow-ups
    prior_trimmed = prior_context_section[:MAX_PRIOR_CONTEXT_CHARS] if is_followup else ""

    # ── Question-type focus hints ────────────────────────────────────────
    focus_hints = {
        "meta": (
            "FOCUS: The user is asking about dataset structure. "
            "Answer ONLY from the META BLOCK below. Give exact numbers. "
            "Do NOT mention previous questions or answers."
        ),
        "stat": (
            "FOCUS: Statistical question. Use the Temperature stats line "
            "(mean, median, std, min, max, range) from DATASET METADATA. "
            "Show the exact numbers from the data. Do NOT mention previous answers."
        ),
        "rank": (
            f"FOCUS: Ranking question asking for top {top_n}. "
            f"Use ONLY the RANKING BLOCK below — it has exactly {top_n} rows sorted correctly. "
            f"List ALL {top_n} entries in order — do NOT truncate the list. "
            "Do NOT reference or repeat any previous answer."
        ),
        "compare": (
            "FOCUS: Comparison question. "
            "Use ONLY the COMPARISON BLOCK below — it contains side-by-side tables and calculated diffs. "
            "Format your answer as a clear side-by-side comparison with all metrics. "
            "Do NOT reference or repeat any previous answer."
        ),
        "trend": (
            "FOCUS: Trend question. Use the 'Overall yearly trend' line and "
            "'Yearly average temperatures' table from DATASET METADATA, "
            "plus TemperatureChange and WarmingTrendPerYear from country/region details. "
            "Do NOT repeat previous answers."
        ),
        "filter": (
            "FOCUS: Filtering question. List all matching countries/regions "
            "from the TARGETED MATCH block, sorted by the metric the user cares about."
        ),
        "summary": (
            "FOCUS: Summary or profile question. Provide a structured overview "
            "covering average temperature, warming trend, volatility, and notable "
            "historical data points for all matched entities."
        ),
        "general": (
            "FOCUS: General climate question. Use all available sections "
            "to give a comprehensive, well-structured answer with specific numbers."
        ),
    }
    focus = focus_hints.get(q_type, focus_hints["general"])

    # ── System prompt — structured by question type ─────────────────────
    # FIX-1: prior context injected ONLY for follow-ups, and clearly gated
    if q_type == "meta":
        data_block = f"""
══════════════════════════════════════════════════════════════
META BLOCK (dataset structure — use ONLY this)
══════════════════════════════════════════════════════════════
{meta_block}
"""
    elif q_type == "rank":
        data_block = f"""
══════════════════════════════════════════════════════════════
RANKING BLOCK (top {top_n} — use ONLY this for the answer)
══════════════════════════════════════════════════════════════
{ranking_block}
"""
    elif q_type == "compare":
        data_block = f"""
══════════════════════════════════════════════════════════════
COMPARISON BLOCK (use ONLY this for the answer)
══════════════════════════════════════════════════════════════
{comparison_block}
"""
    else:
        data_block = f"""
══════════════════════════════════════════════════════════════
COMPLETE DATASET FACTS
══════════════════════════════════════════════════════════════
{full_facts_trimmed}
{dynamic_section}
{supplement_trimmed}

══════════════════════════════════════════════════════════════
REASONING CHAIN: {chain.get("chain_name", "climate")}
{evidence_trimmed}
"""

    # FIX-5: Only include prior context section when it's actually a follow-up
    followup_block = ""
    if is_followup and prior_trimmed:
        followup_block = f"""
══════════════════════════════════════════════════════════════
PRIOR CONTEXT (follow-up continuity)
══════════════════════════════════════════════════════════════
{prior_trimmed}
"""

    system_prompt = f"""You are EcoLens Climate Intelligence AI — a precise, data-driven climate assistant.

QUESTION TYPE: {q_type.upper()}
{focus}

{data_block}
{followup_block}

ANSWER RULES:
1. NEVER invent numbers. Every figure must come from the data above.
2. Answer ONLY the current question — do NOT summarise or reference previous answers unless follow-up.
3. DATASET METADATA has exact counts — use those for "how many" questions.
4. For top-N ranking questions, list ALL N entries without truncating.
5. For comparison questions, use the COMPARISON BLOCK data with interpretation.
6. For trend questions, cite the yearly trend table and per-year warming values.
7. For follow-up questions, build on PRIOR CONTEXT rather than restarting.
8. Be precise — use decimal places as given in the data.
9. End your response with exactly: Source: Historical Climate Dataset

OUTPUT FORMAT RULES (the frontend renders your output as rich HTML):
- Use ## for every major section heading (e.g. ## Overview, ## Rankings, ## Key Findings)
- Use **Label**: value on its own line for every key metric (e.g. **Average Temperature**: 25.11 C)
- For ranking lists use a markdown table: | # | Country | Region | Value |
- For comparison answers use a markdown table with one column per entity, rows for each metric
- For bullet findings use "- " prefix under a ## Key Findings or ## Observations section
- Do NOT use raw "- " bullets outside of a ## section heading
- Do NOT use triple asterisks or deeply nested bullets
"""

    answer = _llm_answer(system_prompt, question, history)

    # Graceful fallback if LLM is unavailable
    if answer.startswith("AI temporarily unavailable"):
        answer = _fallback_climate_answer(
            question=enriched_question,
            previous_answer=analysis_state.get("last_answer", "") if is_followup else "",
            is_followup=is_followup,
            main_rows=matched_main,
            country_rows=matched_countries,
            region_rows=matched_regions,
        )

    new_state = _state_for_answer(
        question, answer, "climate",
        followup=is_followup,
        chain_name=chain.get("chain_name"),
        evidence=chain.get("evidence"),
    )

    return {
        "response": answer,
        "formatted_response": _fmt_response(answer, CLIMATE_SOURCE) if _FMT else answer,
        "analysis_state": new_state,
        "source": CLIMATE_SOURCE,
        "data_source": CLIMATE_SOURCE,
        "success": True,
    }


# =========================
# FIX-2: META ANSWER BLOCK BUILDER
# =========================

def _build_meta_answer_block(question: str, main_df, country_df, region_df) -> str:
    """
    Build a tight, fact-only block for meta/debug questions like
    "how many countries", "how many regions", "what columns", "date range".
    FIX-2: previously the LLM received the full noisy prompt and hallucinated;
            now it gets only the exact facts it needs.
    """
    q = str(question).lower()
    lines = []

    df = main_df if main_df is not None and not getattr(main_df, "empty", True) else pd.DataFrame()

    if not df.empty:
        lines.append(f"Total records in dataset: {len(df)}")
        lines.append(f"Columns: {', '.join(map(str, df.columns.tolist()))}")
        lines.append(f"Number of columns: {len(df.columns)}")

        if "Country" in df.columns:
            countries = sorted(df["Country"].dropna().astype(str).unique().tolist())
            lines.append(f"Number of unique countries: {len(countries)}")
            if "list" in q or "all countries" in q or "countries are" in q:
                lines.append(f"All countries: {', '.join(countries)}")

        if "Region" in df.columns:
            regions = sorted(df["Region"].dropna().astype(str).unique().tolist())
            lines.append(f"Number of unique regions: {len(regions)}")
            if "list" in q or "all regions" in q or "regions are" in q:
                lines.append(f"All regions: {', '.join(regions)}")

        if "Year" in df.columns:
            years = pd.to_numeric(df["Year"], errors="coerce").dropna()
            if not years.empty:
                lines.append(f"Year range: {int(years.min())} to {int(years.max())}")
                lines.append(f"Number of unique years: {int(years.nunique())}")

        if "AvgTemperature" in df.columns:
            temp = pd.to_numeric(df["AvgTemperature"], errors="coerce").dropna()
            if not temp.empty:
                lines.append(
                    f"Temperature stats — mean: {temp.mean():.3f} C, median: {temp.median():.3f} C, "
                    f"min: {temp.min():.3f} C, max: {temp.max():.3f} C, std: {temp.std():.3f} C"
                )

    # Also pull from country_trends if available
    if country_df is not None and not getattr(country_df, "empty", True):
        if "Country" in country_df.columns:
            ct_countries = sorted(country_df["Country"].dropna().astype(str).unique().tolist())
            lines.append(f"Countries in country_trends table: {len(ct_countries)}")

    if region_df is not None and not getattr(region_df, "empty", True):
        if "Region" in region_df.columns:
            rt_regions = sorted(region_df["Region"].dropna().astype(str).unique().tolist())
            lines.append(f"Regions in regional_trends table: {len(rt_regions)}")
            lines.append(f"Region names: {', '.join(rt_regions)}")

    return "\n".join(lines) if lines else "Dataset metadata not available."


def _fallback_climate_answer(
    question="",
    previous_answer="",
    is_followup=False,
    main_rows=None,
    country_rows=None,
    region_rows=None,
    fastest_rows=None,
    hottest_rows=None,
) -> str:
    q = str(question).lower()
    parts = []

    country_records = _format_records(country_rows, max_rows=3)
    yearly_records = _format_records(main_rows, max_rows=3)
    region_records = _format_records(region_rows, max_rows=3)
    fastest_records = _format_records(fastest_rows, max_rows=3)
    hottest_records = _format_records(hottest_rows, max_rows=3)

    # FIX-1: only include previous answer in fallback for genuine follow-ups
    if is_followup and previous_answer:
        parts.append("Continuing from the previous result, I am using the same CSV climate context.")

    wants_region_summary = any(word in q for word in ["region", "regional", "summarize", "summary", "trend", "change"])

    if region_records and wants_region_summary:
        parts.append(_regional_detailed_summary(region_records[0], country_records))

    if yearly_records and not wants_region_summary:
        summary = _yearly_temperature_summary(yearly_records[0])
        if summary:
            parts.append(summary)

    if country_records and not (region_records and wants_region_summary):
        parts.append(_country_trend_summary(country_records[0]))

    if region_records and not wants_region_summary:
        parts.append(_regional_trend_summary(region_records[0]))

    if country_records and region_records and not wants_region_summary:
        comparison = _country_region_comparison(country_records[0], region_records[0])
        if comparison:
            parts.append(comparison)

    if any(word in q for word in ["fastest", "most warming", "highest warming"]) and fastest_records:
        facts = [f for f in [_format_fact_row(r) for r in fastest_records] if f]
        if facts:
            parts.append("Fastest warming regions:\n- " + "\n- ".join(facts))

    if any(word in q for word in ["hottest", "warmest", "highest temperature"]) and hottest_records:
        facts = [f for f in [_format_fact_row(r) for r in hottest_records] if f]
        if facts:
            parts.append("Hottest countries:\n- " + "\n- ".join(facts))

    if any(word in q for word in ["why", "reason", "explain", "how"]):
        parts.append(
            "The likely explanation in this dataset is long-term warming: the country and regional rows show positive "
            "temperature change and positive warming trend values. Volatility tells you how uneven the year-to-year "
            "changes are, while the warming trend shows the broader direction."
        )

    if not parts and yearly_records:
        facts = [f for f in [_format_fact_row(r) for r in yearly_records] if f]
        if facts:
            parts.append("Matching yearly rows:\n- " + "\n- ".join(facts))

    if not parts:
        parts.append("No matching climate rows were found. Try using a country, region, or ranking phrase from the dataset.")

    parts.insert(0, "Based on the CSV climate dataset:")
    parts.append("Source: Historical Climate Dataset")
    return "\n\n".join(parts)


def _fmt_number(value, suffix="", decimals=3):
    try:
        if value is None or pd.isna(value):
            return None
        return f"{float(value):.{decimals}f}{suffix}"
    except Exception:
        return str(value) if value not in (None, "") else None


def _country_trend_summary(row):
    country = row.get("Country", "This country")
    region = row.get("Region")
    latest = _fmt_number(row.get("LatestTemperature"), " C", 2)
    avg = _fmt_number(row.get("AverageTemperature"), " C", 2)
    change = _fmt_number(row.get("TemperatureChange"), " C", 2)
    trend = _fmt_number(row.get("WarmingTrendPerYear"), " C/year", 4)
    hottest_year = row.get("HottestYear")
    hottest_temp = _fmt_number(row.get("HottestTemperature"), " C", 2)
    volatility = _fmt_number(row.get("VolatilityScore"), "", 3)

    opener = f"For {country}" + (f" in {region}" if region else "")
    sentences = []
    if latest:
        sentences.append(f"{opener}, the latest temperature in the CSV is {latest}.")
    if change and trend:
        sentences.append(f"The dataset shows a total warming of {change}, with a trend of {trend}.")
    elif change:
        sentences.append(f"The dataset shows a total warming of {change}.")
    if avg:
        sentences.append(f"Its average temperature across the dataset period is {avg}.")
    if hottest_year and hottest_temp:
        sentences.append(f"The hottest recorded year is {hottest_year}, at {hottest_temp}.")
    if volatility:
        sentences.append(f"The volatility score is {volatility}.")
    return " ".join(sentences) if sentences else _format_fact_row(row)


def _yearly_temperature_summary(row):
    country = row.get("Country", "This country")
    region = row.get("Region")
    year = row.get("Year")
    avg_temp = _fmt_number(row.get("AvgTemperature"), " C", 2)
    temp_change = _fmt_number(row.get("TempChange"), " C", 2)
    total_change = _fmt_number(row.get("TotalTempChange"), " C", 2)
    regional_avg = _fmt_number(row.get("RegionalAverageTemperature"), " C", 2)

    label = f"{country}" + (f" in {region}" if region else "")
    sentences = []
    if year and avg_temp:
        sentences.append(f"For {label}, the CSV yearly temperature for {year} is {avg_temp}.")
    elif avg_temp:
        sentences.append(f"For {label}, the matching CSV yearly temperature is {avg_temp}.")
    if temp_change:
        sentences.append(f"The year-to-year temperature change is {temp_change}.")
    if total_change:
        sentences.append(f"The total change from the dataset baseline is {total_change}.")
    if regional_avg:
        sentences.append(f"The matching regional average temperature is {regional_avg}.")
    return " ".join(sentences) if sentences else ""


def _regional_trend_summary(row):
    region = row.get("Region", "the matching region")
    first = _fmt_number(row.get("FirstRegionalTemperature"), " C", 2)
    latest = _fmt_number(row.get("LatestRegionalTemperature"), " C", 2)
    change = _fmt_number(row.get("RegionalTemperatureChange"), " C", 2)
    trend = _fmt_number(row.get("RegionalWarmingTrendPerYear"), " C/year", 4)
    volatility = _fmt_number(row.get("RegionalVolatilityScore"), "", 3)

    sentences = []
    if first and latest:
        sentences.append(f"At the regional level, {region} rose from {first} to {latest}.")
    if change and trend:
        sentences.append(f"That is a regional change of {change}, with a trend of {trend}.")
    elif change:
        sentences.append(f"That is a regional change of {change}.")
    if volatility:
        sentences.append(f"The regional volatility score is {volatility}.")
    return " ".join(sentences) if sentences else _format_fact_row(row)


def _regional_detailed_summary(region_row, country_records=None):
    region = region_row.get("Region", "the matching region")
    first_year = region_row.get("FirstYear")
    latest_year = region_row.get("LatestYear")
    first = _fmt_number(region_row.get("FirstRegionalTemperature"), " C", 2)
    latest = _fmt_number(region_row.get("LatestRegionalTemperature"), " C", 2)
    average = _fmt_number(region_row.get("AverageRegionalTemperature"), " C", 2)
    change = _fmt_number(region_row.get("RegionalTemperatureChange"), " C", 2)
    trend = _fmt_number(region_row.get("RegionalWarmingTrendPerYear"), " C/year", 4)
    volatility = _fmt_number(region_row.get("RegionalVolatilityScore"), "", 3)

    paragraphs = []
    if first and latest:
        period = f"from {first_year} to {latest_year}" if first_year and latest_year else "across the dataset period"
        paragraphs.append(
            f"{region}'s regional temperature trend is upward {period}: "
            f"the regional average moves from {first} to {latest}."
        )
    metric_bits = []
    if change:
        metric_bits.append(f"total regional change is {change}")
    if trend:
        metric_bits.append(f"warming trend is {trend}")
    if average:
        metric_bits.append(f"period average is {average}")
    if metric_bits:
        paragraphs.append(
            "The main metrics show that " + ", ".join(metric_bits) + ". "
            "The total change describes start-to-end warming; the per-year trend describes the average yearly direction."
        )
    if volatility:
        paragraphs.append(
            f"The regional volatility score is {volatility}. A higher value means the year-to-year path is less smooth."
        )
    examples = []
    for row in (country_records or []):
        country = row.get("Country")
        if country:
            details = []
            c_change = _fmt_number(row.get("TemperatureChange"), " C", 2)
            c_trend = _fmt_number(row.get("WarmingTrendPerYear"), " C/year", 4)
            c_latest = _fmt_number(row.get("LatestTemperature"), " C", 2)
            if c_change:
                details.append(f"change {c_change}")
            if c_trend:
                details.append(f"trend {c_trend}")
            if c_latest:
                details.append(f"latest {c_latest}")
            examples.append(f"{country}: " + ", ".join(details) if details else str(country))
    if examples:
        paragraphs.append("Country examples inside this region:\n- " + "\n- ".join(examples))
    if change or trend:
        paragraphs.append(
            f"Overall, the CSV supports a warming interpretation for {region}: "
            "the latest regional temperature is above the first recorded value and the trend is positive."
        )
    return "\n\n".join(paragraphs) if paragraphs else _regional_trend_summary(region_row)


def _country_region_comparison(country_row, region_row):
    country = country_row.get("Country", "The country")
    region = region_row.get("Region", country_row.get("Region", "its region"))
    country_change = _fmt_number(country_row.get("TemperatureChange"), " C", 2)
    region_change = _fmt_number(region_row.get("RegionalTemperatureChange"), " C", 2)
    country_trend = _fmt_number(country_row.get("WarmingTrendPerYear"), " C/year", 4)
    region_trend = _fmt_number(region_row.get("RegionalWarmingTrendPerYear"), " C/year", 4)

    if not country_change and not region_change:
        return ""
    if country_change and region_change:
        return (
            f"Compared with {region}, {country}'s total warming is {country_change}, "
            f"while the regional change is {region_change}. "
            f"Per-year trends: {country_trend or 'N/A'} for {country}, "
            f"{region_trend or 'N/A'} for {region}."
        )
    return (
        f"Country-level change: {country_change or 'N/A'}, "
        f"regional change: {region_change or 'N/A'}."
    )


# =========================
# HYBRID: WEATHER + CLIMATE
# =========================

def _answer_weather_and_climate(
    question: str,
    default_location=None,
    data=None,
    history=None,
    analysis_state=None,
):
    data = data or {}
    history = history or []
    analysis_state = analysis_state or {}

    top_n = _parse_top_n(question, default=10)
    is_followup = _is_followup_question(question)

    # FIX-5: Only enrich for genuine follow-ups
    enriched_question = question
    prior_block = ""
    if is_followup and analysis_state.get("last_question"):
        prior_block = (
            f"PREVIOUS QUESTION:\n{analysis_state.get('last_question')}\n\n"
            f"PREVIOUS ANSWER:\n{analysis_state.get('last_answer', '')[:MAX_PRIOR_CONTEXT_CHARS]}\n\n"
        )
        enriched_question = (
            f"PREVIOUS QUESTION:\n{analysis_state.get('last_question')}\n\n"
            f"PREVIOUS ANSWER:\n{analysis_state.get('last_answer', '')[:400]}\n\n"
            f"FOLLOW-UP QUESTION:\n{question}"
        )

    location = extract_location(question) or default_location
    if not location:
        return {
            "response": "Please include a valid city name (e.g., Chennai, London, Tokyo).",
            "analysis_state": analysis_state,
            "source": "Both",
            "data_source": "Both",
            "success": False,
        }

    weather = None
    try:
        weather_response = get_current_weather(location)
        if isinstance(weather_response, dict):
            if "data" in weather_response:
                weather = weather_response["data"]
            elif "response" in weather_response:
                weather = weather_response["response"]
            elif "location" in weather_response or "current" in weather_response:
                weather = weather_response
        weather = normalize_weather(weather) or {}
    except Exception as exc:
        logger.warning("WEATHER ERROR: %s", exc)
        weather = {}

    weather_evidence = "Live weather unavailable."
    try:
        if weather:
            place = weather.get("location", {})
            current = weather.get("current", {})
            today = weather.get("today", {})
            location_label = ", ".join(
                part for part in [place.get("name"), place.get("admin1"), place.get("country")] if part
            )
            weather_evidence = (
                f"Location: {location_label}\n"
                f"Temperature: {current.get('temperature_c')} °C\n"
                f"Feels Like: {current.get('feels_like_c')} °C\n"
                f"Condition: {current.get('condition')}\n"
                f"Humidity: {current.get('humidity_percent')}%\n"
                f"Wind Speed: {current.get('wind_speed_kmh')} km/h\n"
                f"Today Min: {today.get('min_temperature_c')} °C\n"
                f"Today Max: {today.get('max_temperature_c')} °C\n"
                f"Rainfall: {today.get('rain_sum_mm')} mm"
            )
    except Exception as exc:
        weather_evidence = f"Weather parsing error: {str(exc)}"

    try:
        yearly_df = data.get("main")
        country_trends = data.get("country_trends")
        regional_trends = data.get("regional_trends")
        hottest_countries = data.get("hottest_countries")
        fastest_regions = data.get("fastest_regions")
    except Exception:
        return {
            "response": "Climate dataset unavailable.",
            "analysis_state": analysis_state,
            "source": "Climate Dataset",
            "data_source": "Climate Dataset",
            "success": False,
        }

    country_name = weather.get("location", {}).get("country")
    country_context = "No matching country climate data."

    try:
        if country_name and country_trends is not None and not country_trends.empty:
            name_lower = country_name.lower()
            mask = pd.Series(False, index=country_trends.index)
            for col in country_trends.columns:
                mask = mask | country_trends[col].astype(str).str.lower().str.contains(
                    re.escape(name_lower), na=False
                )
            matched = country_trends[mask]
            if not matched.empty:
                country_context = matched.head(10).to_string(index=False)
    except Exception as exc:
        logger.warning("COUNTRY MATCH ERROR: %s", exc)

    climate_context = (
        f"\nMAIN DATASET:\n{safe_preview(yearly_df)}\n\n"
        f"COUNTRY TRENDS:\n{safe_preview(country_trends, top_n)}\n\n"
        f"REGIONAL TRENDS:\n{safe_preview(regional_trends, top_n)}\n\n"
        f"HOTTEST COUNTRIES:\n{safe_preview(hottest_countries, top_n)}\n\n"
        f"FASTEST WARMING REGIONS:\n{safe_preview(fastest_regions, top_n)}"
    )

    def get_dataset_context() -> str:
        try:
            import json
            schema_path = get_processed_path("schema.json")
            with open(schema_path) as f:
                schema = json.load(f)
            source = get_active_source()
            label = "the default climate dataset" if source == "default" else "a user-uploaded dataset"
            cols = ", ".join(schema.get("columns", []))
            cats = ", ".join(schema.get("categorical_columns", []))
            nums = ", ".join(schema.get("numeric_columns", []))
            rows = schema.get("row_count", 0)
            return (
                f"\n\nACTIVE DATASET INFO: You are analyzing {label} with {rows} rows. "
                f"All columns: {cols}. "
                f"Text/category columns: {cats}. "
                f"Numeric columns: {nums}. "
                f"Answer questions using these exact column names."
            )
        except Exception:
            return ""

    # FIX-1: only include prior_block when is_followup
    prior_section = prior_block if is_followup else ""

    dataset_context = get_dataset_context()
    system_prompt = f"""You are EcoLens Climate Intelligence AI — a precise, data-driven climate assistant.
    {dataset_context}

IMPORTANT: Answer ONLY the current question. Do NOT repeat or summarise previous answers unless this is a follow-up.
{prior_section}
LIVE WEATHER DATA:
{weather_evidence}

MATCHED COUNTRY CLIMATE DATA:
{country_context}

CSV CLIMATE DATA:
{climate_context[:MAX_FACTS_CHARS]}

RULES:
1. Use BOTH live weather and climate CSV data.
2. For follow-up questions only — build on prior context.
3. If exact city data is unavailable, use country or regional insights.
4. Never invent values.
5. End with: Source: Live Weather + Climate Dataset

OUTPUT FORMAT RULES:
- ## Current Weather - <location> section: use **Label**: value lines for each weather field 
- ## Climate Context section: use **Label**: value for key climate metrics
- ## Key Observations section: 2-4 bullet points comparing live vs historical
- Use markdown tables only for multi-row data (rankings, comparisons)
{dataset_context}"""
    answer = _llm_answer(system_prompt, enriched_question, history)

    if answer.startswith("AI temporarily unavailable"):
        answer = (
            f"Using live weather and climate data,\n\n"
            f"LIVE WEATHER:\n{weather_evidence}\n\n"
            f"CLIMATE CONTEXT:\n{country_context}\n\n"
            f"Source: Live Weather + Climate Dataset"
        )

    new_state = {
        "last_question": question,
        "last_answer": answer[:MAX_LAST_ANSWER_CHARS],
        "followup": is_followup,
        "last_source": "hybrid",
        "last_location": location,
        "last_weather": weather or {},
    }

    return {
        "response": answer,
        "formatted_response": _fmt_response(answer, "Live Weather + Climate Dataset") if _FMT else answer,
        "analysis_state": new_state,
        "source": "Live Weather + Climate Dataset",
        "data_source": "Live Weather + Climate Dataset",
        "success": True,
    }


# =========================
# HELPERS
# =========================

def _format_records(df, max_rows=3):
    if df is None or getattr(df, "empty", True):
        return []
    try:
        return df.head(max_rows).to_dict(orient="records")
    except Exception:
        return []


def _format_fact_row(row):
    useful_keys = [
        "Country", "Region", "LatestTemperature", "HottestYear", "HottestTemperature",
        "AverageTemperature", "TemperatureChange", "WarmingTrendPerYear", "VolatilityScore",
        "FirstRegionalTemperature", "LatestRegionalTemperature", "RegionalTemperatureChange",
        "RegionalAverageTemperature", "RegionalWarmingTrendPerYear", "RegionalVolatilityScore",
        "Year", "AvgTemperature", "TempChange", "TotalTempChange",
    ]
    facts = []
    for key in useful_keys:
        value = row.get(key)
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        facts.append(f"{key}: {value}")
    return "; ".join(facts)


# =========================
# WEATHER FORMATTER
# =========================

def _format_weather_answer(weather: dict) -> str:
    if not weather:
        return "Weather data unavailable."
    place = weather.get("location", {})
    current = weather.get("current", {})
    today = weather.get("today", {})
    location_label = ", ".join(
        part for part in [place.get("name"), place.get("admin1"), place.get("country")] if part
    )
    return (
        f"Weather Analysis\n\n"
        f"Location: {location_label}\n"
        f"Temperature: {current.get('temperature_c', 'N/A')}°C\n"
        f"Feels like: {current.get('feels_like_c', 'N/A')}°C\n"
        f"Condition: {current.get('condition', 'N/A')}\n"
        f"Humidity: {current.get('humidity_percent', 'N/A')}%\n"
        f"Wind: {current.get('wind_speed_kmh', 'N/A')} km/h\n\n"
        f"Forecast:\n"
        f"Min: {today.get('min_temperature_c', 'N/A')}°C\n"
        f"Max: {today.get('max_temperature_c', 'N/A')}°C\n"
        f"Rain: {today.get('rain_sum_mm', 'N/A')} mm\n"
    )


# =========================
# CLIMATE EVIDENCE BUILDER
# =========================

def _build_climate_evidence(question, yearly_df, country_trends, regional_trends):
    q = str(question).lower()
    year_match = re.search(r"(19|20)\d{2}", q)
    year = int(year_match.group()) if year_match else None

    def safe(df):
        return df.copy() if df is not None else pd.DataFrame()

    country_df = safe(country_trends)
    region_df = safe(regional_trends)
    yearly_df = safe(yearly_df)
    parts = []

    matched_countries = []
    if not country_df.empty and "Country" in country_df.columns:
        for country in country_df["Country"].dropna().astype(str).unique():
            if re.search(rf"\b{re.escape(country.lower())}\b", q):
                matched_countries.append(country)

    matched_regions = []
    if not region_df.empty and "Region" in region_df.columns:
        for region in region_df["Region"].dropna().astype(str).unique():
            if re.search(rf"\b{re.escape(region.lower())}\b", q):
                matched_regions.append(region)

    for country in matched_countries:
        cdf = country_df[country_df["Country"].astype(str).str.lower() == country.lower()].copy()
        if year and "Year" in cdf.columns:
            cdf["Year"] = pd.to_numeric(cdf["Year"], errors="coerce")
            cdf = cdf[cdf["Year"] == year]
        if not cdf.empty:
            parts.append(f"COUNTRY DATA ({country}):\n" + cdf.head(50).to_string(index=False))

    for region in matched_regions:
        rdf = region_df[region_df["Region"].astype(str).str.lower() == region.lower()].copy()
        if year and "Year" in rdf.columns:
            rdf["Year"] = pd.to_numeric(rdf["Year"], errors="coerce")
            rdf = rdf[rdf["Year"] == year]
        if not rdf.empty:
            parts.append(f"REGIONAL DATA ({region}):\n" + rdf.head(50).to_string(index=False))

    if not parts:
        if not country_df.empty:
            sort_col = next(
                (c for c in ["AverageTemperature", "TemperatureChange", "AvgTemperature"]
                 if c in country_df.columns),
                None,
            )
            if sort_col:
                ranked = country_df.copy()
                ranked["__s"] = pd.to_numeric(ranked[sort_col], errors="coerce")
                ranked = ranked.dropna(subset=["__s"]).sort_values("__s", ascending=False).drop(columns=["__s"])
                parts.append("COUNTRY RANKINGS (all, sorted by temperature desc):\n"
                              + ranked.head(50).to_string(index=False))
            else:
                parts.append("COUNTRY DATA (overview):\n" + country_df.head(50).to_string(index=False))
        if not region_df.empty:
            parts.append("REGIONAL DATA (overview):\n" + region_df.head(20).to_string(index=False))
        if not yearly_df.empty:
            parts.append("YEARLY DATA (sample):\n" + yearly_df.head(20).to_string(index=False))

    return {"text": "\n\n".join(parts) if parts else ""}


def _matching_rows(df, column, query):
    if df is None or df.empty or column not in df.columns or not query:
        return df.head(0) if df is not None else pd.DataFrame()
    try:
        q = re.sub(r"[^a-zA-Z0-9\s]", " ", str(query).lower().strip())
        words = q.split()
        stop_words = {
            "what", "is", "the", "temperature", "weather", "climate", "trend",
            "in", "at", "of", "for", "show", "tell", "me", "about", "could",
            "be", "reason", "why", "how", "today", "current",
        }
        keywords = [w for w in words if w not in stop_words] or words
        city_country_map = {
            "chennai": "india", "mumbai": "india", "delhi": "india",
            "bangalore": "india", "kolkata": "india", "tokyo": "japan",
            "london": "united kingdom", "paris": "france",
            "new york": "united states", "sydney": "australia",
        }
        expanded = []
        for kw in keywords:
            expanded.append(kw)
            if kw in city_country_map:
                expanded.append(city_country_map[kw])
        series = df[column].astype(str).str.lower().fillna("")
        scores = [sum(1 for kw in expanded if kw in row) for row in series]
        matched = df.copy()
        matched["_score"] = scores
        matched = matched[matched["_score"] > 0].sort_values("_score", ascending=False).drop(columns=["_score"])
        return matched.head(20) if not matched.empty else df.head(10)
    except Exception:
        return df.head(10)


def _match_country_for_weather(country_trends, country_name):
    if country_trends is None or country_trends.empty or not country_name or "Country" not in country_trends.columns:
        return country_trends.iloc[0:0] if country_trends is not None else None
    try:
        df = country_trends.copy()
        name = " ".join(re.sub(r"[^a-zA-Z0-9\s]", " ", str(country_name).lower()).split())
        aliases = {
            "usa": "united states", "us": "united states", "america": "united states",
            "uk": "united kingdom", "britain": "united kingdom", "england": "united kingdom",
            "uae": "united arab emirates", "south korea": "korea", "north korea": "korea",
            "russian federation": "russia", "people s republic of china": "china",
            "czech republic": "czechia", "viet nam": "vietnam",
        }
        normalized_name = aliases.get(name, name)
        country_series = df["Country"].astype(str).str.lower().fillna("")
        scores = []
        for row_country in country_series:
            if normalized_name == row_country:
                scores.append(100)
            elif normalized_name in row_country:
                scores.append(50)
            else:
                scores.append(sum(10 for w in normalized_name.split() if w in row_country))
        df["_score"] = scores
        matched = df[df["_score"] > 0].sort_values("_score", ascending=False).drop(columns=["_score"])
        return matched.head(20) if not matched.empty else country_trends.iloc[0:0]
    except Exception:
        return country_trends.iloc[0:0] 