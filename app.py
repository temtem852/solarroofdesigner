# =========================================================
# Solar Rooftop Designer — All-in-One (Production Ready)
# =========================================================

import os, re, json
import numpy as np
from serpapi import GoogleSearch
import google.generativeai as genai
from fpdf import FPDF
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
from dotenv import load_dotenv


import streamlit as st

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from datetime import datetime
import io



load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
SPREADSHEET_KEY = os.getenv("SPREADSHEET_KEY")
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")

LLM_PROVIDER = None

if GEMINI_KEY:
    LLM_PROVIDER = "gemini"
elif OPENAI_KEY:
    LLM_PROVIDER = "openai"


# =========================================================
# APP CONFIG
# =========================================================
st.set_page_config(
    page_title="Solar Rooftop Designer",
    page_icon="🔆",
    layout="wide"
)

st.title(" Solar Rooftop Designer ")


from fpdf import FPDF
from fpdf.enums import XPos, YPos
from datetime import datetime





# =========================================================
# CONFIG
# =========================================================



# ต้องตรงกับชื่อ tab จริงใน Google Sheets
TAB_KEYWORDS = {
    "Solar_Panels": [
        "panel", "solar panel", "pv module", "module",
        "mono", "perc", "topcon", "bifacial", "vertex", "tiger"
    ],
    "Inverters": [
        "inverter", "string inverter", "hybrid inverter",
        "on-grid", "off-grid", "mppt", "sungrow", "growatt", "huawei"
    ],
    "Batteries": [
        "battery", "lithium", "lifepo4", "storage", "bms"
    ],
    "Accessories": [
        "mount", "rail", "clamp", "mc4",
        "dc cable", "ac cable", "combiner"
    ]
}

DEFAULT_TAB = "Accessories"
if "TH" not in pdfmetrics.getRegisteredFontNames():
    pdfmetrics.registerFont(TTFont("TH", "THSarabunNew.ttf"))
    pdfmetrics.registerFont(TTFont("TH-B", "THSarabunNew-Bold.ttf"))

# =========================================================
# CONNECT GOOGLE SHEETS
# =========================================================

@st.cache_resource
def connect_spreadsheet():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    creds = Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=scopes,
    )

    client = gspread.authorize(creds)
    return client.open_by_key(SPREADSHEET_KEY)

# =========================================================
# AUTO DETECT WORKSHEET
# =========================================================

def detect_worksheet_from_text(text: str, spreadsheet):
    text = text.lower()

    for sheet_name, keywords in TAB_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                try:
                    return spreadsheet.worksheet(sheet_name)
                except gspread.exceptions.WorksheetNotFound:
                    st.warning(f"⚠️ ไม่พบ tab: {sheet_name}")

    # fallback
    return spreadsheet.worksheet(DEFAULT_TAB)

# =========================================================
# LOAD DATABASE FROM WORKSHEET
# =========================================================

def load_db(worksheet):
    records = worksheet.get_all_records()
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)

# =========================================================
# APPEND ROW TO WORKSHEET
# =========================================================

def append_to_sheet(worksheet, row: list):
    worksheet.append_row(
        row,
        value_input_option="USER_ENTERED"
    )

# =========================================================
# HIGH-LEVEL HELPER (ใช้กับ SerpAPI)
# =========================================================

def save_search_result_to_sheet(
    search_query: str,
    brand: str,
    model: str,
    power: float,
    datasheet_url: str,
    source: str = "Google"
):
    spreadsheet = connect_spreadsheet()

    worksheet = detect_worksheet_from_text(
        f"{search_query} {brand} {model}",
        spreadsheet
    )

    append_to_sheet(worksheet, [
        brand,
        model,
        power,
        datasheet_url,
        source,
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        search_query
    ])

    return worksheet.title


import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Detect inverter AC column automatically
# ---------------------------------------------------------
def find_ac_column(df):
    candidates = [
        "Power_kW",
        "AC_kW",
        "Rated Power",
        "AC Power (kW)",
        "AC Power"
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


# ---------------------------------------------------------
# 🔎 LLM Explanation Layer (Auto Fastest Priority)
# ---------------------------------------------------------
def generate_llm_explanation(prompt, GEMINI_KEY=None, OPENAI_KEY=None):

    openai_error = None
    gemini_error = None

    # =====================================================
    # 1Priority: OpenAI (Fast + Stable)
    # =====================================================
    if OPENAI_KEY:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=OPENAI_KEY)

            response = client.chat.completions.create(
                model="gpt-4o-mini",   # ⚡ fastest stable
                messages=[
                    {"role": "system", "content": "You are a professional solar PV engineer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )

            content = response.choices[0].message.content

            if content:
                return content.strip()

        except Exception as e:
            openai_error = str(e)

    # =====================================================
    # Fallback: Gemini
    # =====================================================
    if GEMINI_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_KEY)

            # Try stable models in order
            gemini_models = [
                "models/gemini-1.5-flash",
                "models/gemini-1.5-pro",
                "gemini-pro","models/gemini-2.5-flash"
            ]

            for model_name in gemini_models:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)

                    if hasattr(response, "text") and response.text:
                        return response.text.strip()

                except Exception as inner:
                    gemini_error = str(inner)
                    continue

        except Exception as e:
            gemini_error = str(e)

    # =====================================================
    # Deterministic Fallback
    # =====================================================
    return f"""
AI explanation unavailable.

OpenAI error: {openai_error}
Gemini error: {gemini_error}

System proceeds with deterministic MCDM result only.
"""


# ---------------------------------------------------------
# Gaussian Function
# ---------------------------------------------------------
def gaussian_penalty(x, x0, sigma):
    sigma = max(float(sigma), 1e-6)
    return np.exp(-((x - x0) / sigma) ** 2)
# ---------------------------------------------------------
# Main Hybrid Selection Function
# ---------------------------------------------------------
def ai_select_from_database(
    panels_df,
    inverters_df,
    dc_capacity,
    dc_ac_ratio,
    area,
    GEMINI_KEY=None,
    OPENAI_KEY=None
):

    if panels_df.empty or inverters_df.empty:
        return "⚠️ Database is empty."

    # =====================================================
    # Deterministic MCDM Selection
    # =====================================================

    ac_col = find_ac_column(inverters_df)
    if ac_col is None:
        return "❌ Cannot detect inverter AC power column."

    df_inv = inverters_df.copy()
    df_inv[ac_col] = pd.to_numeric(df_inv[ac_col], errors="coerce")
    df_inv = df_inv.dropna(subset=[ac_col])

    if df_inv.empty:
        return "⚠️ No valid inverter data."

    # ---- DC/AC Ratio
    df_inv["ratio"] = dc_capacity / df_inv[ac_col]

    # ---- Gaussian Scores
    df_inv["score_ratio"] = gaussian_penalty(df_inv["ratio"], 1.1, 0.15)
    df_inv["score_capacity"] = gaussian_penalty(
        df_inv[ac_col], dc_capacity, dc_capacity * 0.2
    )

    # ---- Weighted Sum
    w_ratio = 0.6
    w_capacity = 0.4

    df_inv["total_score"] = (
        w_ratio * df_inv["score_ratio"] +
        w_capacity * df_inv["score_capacity"]
    )

    df_inv = df_inv.sort_values("total_score", ascending=False)

    top_inverters = df_inv.head(3)
    best_inv = top_inverters.iloc[0]

    # =====================================================
    # Panel Selection (Gaussian Power Preference)
    # =====================================================

    df_pan = panels_df.copy()

    if "Pm(W)" in df_pan.columns:
        df_pan["Pm(W)"] = pd.to_numeric(df_pan["Pm(W)"], errors="coerce")
        df_pan["score_power"] = gaussian_penalty(
            df_pan["Pm(W)"], 550, 100
        )
        df_pan = df_pan.sort_values("score_power", ascending=False)

    top_panels = df_pan.head(3)
    best_panel = top_panels.iloc[0]

    # =====================================================
    # Build LLM Prompt
    # =====================================================

    prompt = f"""
You are a solar PV engineer.

Selection method: Gaussian Weighted Multi-Criteria Decision Making.

PROJECT DATA:
DC Capacity = {dc_capacity:.2f} kWp
DC/AC Ratio = {dc_ac_ratio:.2f}
Roof Area = {area:.2f} m²

TOP INVERTERS:
{top_inverters[[ac_col, "ratio", "total_score"]].to_string(index=False)}

TOP PANELS:
{top_panels.head(3).to_string(index=False)}

SELECTED COMPONENTS:
Inverter: {best_inv.get("Brand","")} {best_inv.get("Model","")}
Panel: {best_panel.get("Brand","")} {best_panel.get("Model","")}

Explain briefly why these rank highest based on:
- DC/AC optimization
- Capacity proximity
- Practical engineering suitability

Do not change the selected models.
Keep concise and professional.
"""

    explanation = generate_llm_explanation(
        prompt,
        GEMINI_KEY=GEMINI_KEY,
        OPENAI_KEY=OPENAI_KEY
    )

    # =====================================================
    # Final Output
    # =====================================================

    result = f"""
====================================================
DETERMINISTIC SELECTION 
====================================================

Selected Inverter:
{best_inv.get("Brand","")} {best_inv.get("Model","")}
AC Rating: {best_inv[ac_col]} kW

Selected Panel:
{best_panel.get("Brand","")} {best_panel.get("Model","")}

----------------------------------------------------
AI ENGINEERING EXPLANATION
----------------------------------------------------
{explanation}
"""

    return result





def irr(cashflows, guess=0.1):
    r = guess
    for _ in range(100):
        f = sum(cf / ((1 + r) ** i) for i, cf in enumerate(cashflows))
        df = sum(-i * cf / ((1 + r) ** (i + 1)) for i, cf in enumerate(cashflows))
        if abs(df) < 1e-9:
            break
        r -= f / df
    return r


def get_value(row, *possible_cols, default=None):
    for col in possible_cols:
        if col in row and pd.notna(row[col]):
            return row[col]
    return default


def pick_column(df, *candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# =========================================================
# SIDEBAR
# =========================================================
# ---- INIT STATE  ----
if "run_design" not in st.session_state:
    st.session_state.run_design = False

with st.sidebar.form("pv_design_form"):

    # ---------- LOAD & RESOURCE ----------
    st.header("ข้อมูลโหลดไฟฟ้า")

    st.number_input(
        "พลังงานไฟฟ้าต่อวัน (kWh/day)",
        min_value=0.0,
        value=30.0,
        step=1.0,
        key="E_day"
    )

    st.number_input(
        "ชั่วโมงแสงอาทิตย์ (Peak Sun Hours)",
        min_value=1.0,
        max_value=7.0,
        value=4.5,
        step=0.1,
        key="H_sun"
    )

    st.slider(
        "Performance Ratio (PR)",
        0.6, 0.9, 0.8, 0.01,
        key="PR"
    )

    # ---------- ROOF AREA ----------
    st.header("พื้นที่ติดตั้ง")

    st.number_input(
        "พื้นที่หลังคาใช้งานได้ (m²)",
        min_value=1.0,
        value=50.0,
        step=1.0,
        key="area"
    )

    # ---------- PV MODULE ----------
    st.header("สเปคแผงโซลาร์ (ต่อแผง)")

    st.number_input("Vmp (V)", 10.0, value=41.0, step=0.1, key="Vmp")
    st.number_input("Voc (V)", 10.0, value=50.0, step=0.1, key="Voc")
    st.number_input("Imp (A)", 1.0, value=13.0, step=0.1, key="Imp")
    st.number_input("Isc (A)", 1.0, value=13.5, step=0.1, key="Isc")
    st.number_input("กำลังแผง (Pm, W)", 100, value=550, step=5, key="Pm")

    # ---------- INVERTER ----------
    st.header("สเปคอินเวอร์เตอร์")

    st.number_input(
        "AC Rated Power (W)",
        min_value=1000,
        value=10000,
        step=500,
        key="inv_power_ac"
    )

    st.number_input(
        "DC Max Voltage (V)",
        min_value=300,
        value=1100,
        step=50,
        key="inv_v_dc_max"
    )

    st.number_input(
        "Max Input Current / MPPT (A)",
        min_value=5.0,
        value=25.0,
        step=1.0,
        key="inv_i_sc_max"
    )

    st.number_input(
        "Max PV Power (W)",
        min_value=1000,
        value=13000,
        step=500,
        key="inv_pv_power_max"
    )

    # ---------- ECONOMICS ----------
    st.header("เศรษฐศาสตร์โครงการ")

    st.number_input(
        "ต้นทุนลงทุน (CAPEX, บาท)",
        min_value=0,
        value=350000,
        step=10000,
        key="CAPEX"
    )

    st.number_input(
        "ค่าไฟฟ้า (Tariff, บาท/kWh)",
        min_value=0.0,
        value=4.0,
        step=0.1,
        key="tariff"
    )

    st.number_input(
        "อายุโครงการ (ปี)",
        min_value=1,
        value=25,
        step=1,
        key="years"
    )

    # ---------- CALCULATE BUTTON ----------
    submitted = st.form_submit_button(" Calculate PV System")

# ---- TRIGGER DESIGN RUN ----
if submitted:
    st.session_state.run_design = True







# =========================================================
# DATABASE VIEW (MULTI-TAB)
# =========================================================

spreadsheet = connect_spreadsheet()

st.header("Equipment Database ")

tabs = {
    "Solar Panels": "Solar_Panels",
    "Inverters": "Inverters",
    "Accessories": "Accessories",
}

tab_ui = st.tabs(list(tabs.keys()))

# Initialize session storage
if "panels_db" not in st.session_state:
    st.session_state["panels_db"] = pd.DataFrame()

if "inverters_db" not in st.session_state:
    st.session_state["inverters_db"] = pd.DataFrame()

if "accessories_db" not in st.session_state:
    st.session_state["accessories_db"] = pd.DataFrame()


for ui_tab, sheet_name in zip(tab_ui, tabs.values()):
    with ui_tab:
        try:
            ws = spreadsheet.worksheet(sheet_name)
            df = load_db(ws)

            if df.empty:
                st.info(f"{sheet_name} ยังไม่มีข้อมูล")
            else:
                st.dataframe(df, use_container_width=True)


                if sheet_name == "Solar_Panels":
                    st.session_state["panels_db"] = df

                elif sheet_name == "Inverters":
                    st.session_state["inverters_db"] = df

                elif sheet_name == "Accessories":
                    st.session_state["accessories_db"] = df

        except Exception as e:
            st.error(f"❌ ไม่สามารถโหลดแท็บ {sheet_name}")
            st.caption(str(e))


# =========================================================
#  SERPAPI SEARCH
# =========================================================
st.header(" ค้นหาอุปกรณ์ ")

c1, c2 = st.columns(2)

with c1:
    eq_type = st.selectbox("ประเภทอุปกรณ์ (Type)", ["Solar_Panels", "Inverters"])
    brand   = st.text_input("ยี่ห้อ (Brand)")
    model   = st.text_input("รุ่น (Model)")
    power   = st.number_input("กำลังไฟฟ้า (Power, W)", min_value=0)

with c2:
    query = st.text_input(
        "คำค้นหา (Search query)",
        value=f"{brand} {model} datasheet filetype:pdf".strip()
    )

# -------------------------------------------------
# SEARCH BUTTON
# -------------------------------------------------
if st.button(" Search & Save"):

    if not SERPAPI_KEY:
        st.error("❌ ยังไม่ได้ตั้งค่า SERPAPI_KEY")
        st.stop()

    if not brand or not model:
        st.warning("⚠️ กรุณากรอก Brand และ Model")
        st.stop()

    # -------------------------------------------------
    # SELECT WORKSHEET
    # -------------------------------------------------
    try:
        ws = spreadsheet.worksheet(eq_type)
    except Exception:
        st.error(f"❌ ไม่พบแท็บ {eq_type} ใน Google Sheets")
        st.stop()

    # -------------------------------------------------
    # LOAD EXISTING DATA
    # -------------------------------------------------
    records = ws.get_all_records()
    df_exist = pd.DataFrame(records) if records else pd.DataFrame()

    # -------------------------------------------------
    # SERPAPI GOOGLE SEARCH
    # -------------------------------------------------
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": 10,
    }

    res = GoogleSearch(params).get_dict()

    # -------------------------------------------------
    # COLLECT PDF DATASHEET CANDIDATES
    # -------------------------------------------------
    pdf_candidates = []

    for r in res.get("organic_results", []):
        link    = r.get("link", "")
        title   = r.get("title", "").lower()
        snippet = r.get("snippet", "").lower()

        if link.lower().endswith(".pdf"):
            score = 0
            if "datasheet" in title or "data sheet" in title:
                score += 2
            if "specification" in title:
                score += 1
            if brand.lower() in title:
                score += 1
            if model.lower() in title:
                score += 2

            pdf_candidates.append({
                "title": r.get("title", ""),
                "link": link,
                "score": score,
                "source": r.get("source", "Google"),
            })

    # sort by relevance score
    pdf_candidates = sorted(
        pdf_candidates,
        key=lambda x: x["score"],
        reverse=True
    )

    # -------------------------------------------------
    # SHOW FOUND LINKS
    # -------------------------------------------------
    st.markdown("### Datasheet ที่พบ ")

    if pdf_candidates:
        for i, p in enumerate(pdf_candidates[:3], start=1):
            st.markdown(
                f"**{i}. {p['title']}**  \n"
                f" [เปิด Datasheet PDF]({p['link']})  \n"
                f"แหล่งที่มา (Source): {p['source']}"
            )
    else:
        st.warning("⚠️ ไม่พบ Datasheet PDF ที่ชัดเจน")

    # -------------------------------------------------
    # PICK BEST DATASHEET (AUTO)
    # -------------------------------------------------
    datasheet = ""
    source = "Google"

    if pdf_candidates:
        datasheet = pdf_candidates[0]["link"]
        source = pdf_candidates[0]["source"]

    # -------------------------------------------------
    # DUPLICATE CHECK (Brand + Model)
    # -------------------------------------------------
    if not df_exist.empty and {"Brand", "Model"}.issubset(df_exist.columns):
        dup = df_exist[
            (df_exist["Brand"].str.lower() == brand.lower()) &
            (df_exist["Model"].str.lower() == model.lower())
        ]
        if not dup.empty:
            st.warning("⚠️ อุปกรณ์นี้มีอยู่แล้วในฐานข้อมูล")
            st.dataframe(dup)
            st.stop()

    # -------------------------------------------------
    # APPEND TO GOOGLE SHEET
    # -------------------------------------------------
    ws.append_row([
        brand,                       # Brand
        model,                       # Model
        power,                       # Power (W)
        "",                          # Price
        datasheet,                   # Datasheet URL
        source,                      # Source
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        query
    ], value_input_option="USER_ENTERED")

    st.success(f"✅ บันทึกอุปกรณ์ลงแท็บ {eq_type} เรียบร้อย")
    st.rerun()


import numpy as np
import streamlit as st
from numpy_financial import irr

# =========================================================
#  PV SYSTEM DESIGN
# =========================================================

# ---------------------------------------------------------
# Helper: safe read from session_state
# ---------------------------------------------------------
def ss(key, default=0.0):
    try:
        return float(st.session_state.get(key, default))
    except:
        return default


st.header(" PV System Design | การออกแบบระบบผลิตไฟฟ้าพลังงานแสงอาทิตย์")

# =========================================================
# ⏯ RUN CONTROL
# =========================================================
if not st.session_state.get("run_design", False):
    st.info("⬅️ กรุณากรอกข้อมูลทาง Sidebar แล้วกด **Run PV System Design**")
    st.stop()

## =========================================================
# DESIGN BASIS (ENGINEERING VALIDATION)
# =========================================================
st.markdown("## Design Basis | ข้อมูลตั้งต้น")

E_day = ss("E_day")      # kWh/day
H_sun = ss("H_sun")      # h/day (PSH)
PR    = ss("PR")         # -
area  = ss("area")       # m²

# ---------------------------------------------------------
# BASIC VALIDATION
# ---------------------------------------------------------
if min(E_day, H_sun, PR, area) <= 0:
    st.error("❌ ข้อมูล Load / PSH / PR / Area ต้องมากกว่า 0")
    st.stop()

# ---------------------------------------------------------
# ENGINEERING RANGE CHECK (PVsyst mindset)
# ---------------------------------------------------------
warnings = []

if not (1.0 <= H_sun <= 7.0):
    warnings.append("PSH อยู่นอกช่วงปกติ (1–7 h/day)")

if not (0.65 <= PR <= 0.90):
    warnings.append("PR อยู่นอกช่วงที่พบได้ทั่วไป (0.65–0.90)")

if E_day < 5:
    warnings.append("โหลดไฟฟ้าค่อนข้างต่ำ อาจไม่คุ้มค่าทางเศรษฐศาสตร์")

if area < 10:
    warnings.append("พื้นที่ติดตั้งจำกัด อาจจำกัดขนาดระบบ")

# ---------------------------------------------------------
# DISPLAY WARNINGS (non-blocking)
# ---------------------------------------------------------
for w in warnings:
    st.warning(f"⚠️ {w}")

# ---------------------------------------------------------
# ENGINEERING SUMMARY
# ---------------------------------------------------------
st.info(
    f"""
**Design Inputs Summary**
- Daily energy demand: **{E_day:.1f} kWh/day**
- Peak Sun Hours (PSH): **{H_sun:.2f} h/day**
- Performance Ratio (PR): **{PR:.2f}**
- Available area: **{area:.1f} m²**
"""
)

# =========================================================
# PV CAPACITY SIZING
# =========================================================
st.markdown("## PV Capacity Sizing | คำนวณขนาดระบบ")

P_pv_load = E_day / (H_sun * PR)
P_pv_area = area * 0.20          # ≈ 200 W/m²

P_pv_design = min(P_pv_load, P_pv_area)
E_est_day   = P_pv_design * H_sun * PR

st.markdown(
    f"""
- PV from load: **{P_pv_load:.2f} kWp**
- PV from area: **{P_pv_area:.2f} kWp**

✅ **Design PV Capacity: {P_pv_design:.2f} kWp**  
Estimated Energy: **{E_est_day:.2f} kWh/day**
"""
)



# =========================================================
# PV MODULE (SIDEBAR | ENGINEERING VALIDATION)
# =========================================================
st.markdown("## PV Module | สเปคแผงจากผู้ใช้")

Pm  = ss("Pm")     # W
Vmp = ss("Vmp")    # V
Voc = ss("Voc")    # V
Imp = ss("Imp")    # A
Isc = ss("Isc")    # A

# --- Basic sanity check ---
if min(Pm, Vmp, Voc, Imp, Isc) <= 0:
    st.error("❌ สเปคแผงไม่ครบหรือมีค่าติดลบ")
    st.stop()

# --- Electrical consistency checks (PVsyst-like) ---
Pm_calc = Vmp * Imp

if Pm_calc < 0.9 * Pm or Pm_calc > 1.1 * Pm:
    st.warning(
        f"⚠️ ความไม่สอดคล้องของสเปคแผง\n"
        f"Pm datasheet = {Pm:.0f} W\n"
        f"Vmp × Imp = {Pm_calc:.0f} W\n"
        "→ ตรวจสอบ datasheet อีกครั้ง"
    )

if Voc <= Vmp:
    st.error("❌ Voc ต้องมากกว่า Vmp")
    st.stop()

if Isc <= Imp:
    st.error("❌ Isc ต้องมากกว่า Imp")
    st.stop()

# --- Engineering info for transparency ---
st.info(
    f"""
**Module Electrical Summary**
- Rated Power (Pm): **{Pm:.0f} W**
- Vmp / Imp: **{Vmp:.1f} V / {Imp:.1f} A**
- Voc / Isc: **{Voc:.1f} V / {Isc:.1f} A**
"""
)

# =========================================================
# INVERTER (SIDEBAR)
# =========================================================
st.markdown("## Inverter | สเปคอินเวอร์เตอร์จากผู้ใช้")

inv_ac = ss("inv_power_ac")      # W
inv_v  = ss("inv_v_dc_max")      # V
inv_i  = ss("inv_i_sc_max")      # A
inv_pv = ss("inv_pv_power_max")  # W

# Engineering assumptions (override later if needed)
mppt_count = 1
v_mppt_min = 200
v_mppt_max = 850

if min(inv_ac, inv_v, inv_i, inv_pv) <= 0:
    st.error("❌ สเปค Inverter ไม่ถูกต้อง")
    st.stop()

dc_ac_actual = P_pv_design * 1000 / inv_ac

if dc_ac_actual < 1.0:
    st.warning("⚠️ Inverter ใหญ่เกินไป → Efficiency ต่ำ")
elif dc_ac_actual > 1.35:
    st.warning("⚠️ DC/AC ratio สูง → เสี่ยง clipping")
else:
    st.info("✅ ขนาด Inverter เหมาะสม")

# =========================================================
# STRING DESIGN
# =========================================================
st.markdown("## String Design | ออกแบบจำนวนแผงต่อ String")

sf_voc_cold = 1.20
sf_vmp_hot  = 0.90
sf_current  = 1.25

n_max_voc  = int(inv_v / (Voc * sf_voc_cold))
n_max_mppt = int(v_mppt_max / Vmp)
n_min_mppt = int(np.ceil(v_mppt_min / (Vmp * sf_vmp_hot)))

panels_per_string = min(n_max_voc, n_max_mppt)

if panels_per_string < n_min_mppt:
    st.error("❌ ไม่สามารถจัด String ให้อยู่ใน MPPT window")
    st.stop()

st.info(f"✔ แผงต่อ String: **{panels_per_string} แผง**")

# =========================================================
# STRING QUANTITY (ENGINEERING-GRADE)
# =========================================================
st.markdown("## String Quantity | คำนวณจำนวน String")

# --- Required DC sizing ---
panels_required = int(np.ceil(P_pv_design * 1000 / Pm))
strings_required = int(np.ceil(panels_required / panels_per_string))

# --- Current limit per MPPT ---
I_string = Isc * sf_current

if I_string <= 0:
    st.error("❌ กระแส String ไม่ถูกต้อง")
    st.stop()

strings_per_mppt_max = int(inv_i // I_string)

if strings_per_mppt_max < 1:
    st.error(
        f"❌ Inverter รับกระแสไม่พอ\n"
        f"I_string = {I_string:.1f} A > I_inv = {inv_i:.1f} A"
    )
    st.stop()

strings_max = strings_per_mppt_max * mppt_count
strings_used = min(strings_required, strings_max)

# --- User feedback ---
st.write(
    f"""
- Panels required: **{panels_required} แผง**
- Strings required (ตามโหลด): **{strings_required} string**
- Inverter รองรับได้สูงสุด: **{strings_max} string**
"""
)

if strings_used < strings_required:
    st.warning(
        "⚠️ จำนวน String ถูกจำกัดด้วยกระแส Inverter\n"
        "→ ระบบอาจผลิตไฟได้ไม่เต็มตาม Design PV"
    )
else:
    st.success("✅ จำนวน String เพียงพอตาม Design PV")

# --- DC power check vs inverter ---
dc_power_installed = panels_per_string * strings_used * Pm

if dc_power_installed > inv_pv:
    st.warning(
        f"⚠️ DC Power ติดตั้ง = {dc_power_installed/1000:.2f} kWp "
        f"เกิน Inverter PV Max ({inv_pv/1000:.2f} kWp)"
    )


# =========================================================
# MPPT ALLOCATION
# =========================================================
st.markdown("## MPPT Allocation | การกระจาย String")

remaining = strings_used
for i in range(1, mppt_count + 1):
    s = min(strings_per_mppt_max, remaining)
    remaining -= s
    st.write(f"- MPPT {i}: **{s} string(s)**")

# =========================================================
# FINAL ELECTRICAL CHECK
# =========================================================
st.markdown("## Final Electrical Check | ตรวจสอบขั้นสุดท้าย")

dc_capacity = panels_per_string * strings_used * Pm / 1000
dc_ac_ratio = dc_capacity / (inv_ac / 1000)

Voc_string = panels_per_string * Voc * sf_voc_cold
Vmp_string = panels_per_string * Vmp * sf_vmp_hot

st.success(
    f"""
### ✅ Final System Configuration
- DC Capacity: **{dc_capacity:.2f} kWp**
- DC/AC Ratio: **{dc_ac_ratio:.2f}**
- Voc,string (cold): **{Voc_string:.0f} V**
- Vmpp,string (hot): **{Vmp_string:.0f} V**
"""
)






st.write(st.session_state.get("ai_result", "ยังไม่ได้เรียก AI"))

# =========================================================
# FINANCIAL PERFORMANCE (PVsyst-Grade)
# =========================================================
st.header("Financial Performance | PVsyst-grade Analysis")

# -------------------------------
# USER INPUTS / ASSUMPTIONS
# -------------------------------
CAPEX = float(st.session_state.get("CAPEX", 480_000))   # THB
project_life = int(st.session_state.get("years", 25))

tariff_self = float(st.session_state.get("tariff", 4.0))   # THB/kWh
tariff_export = float(st.session_state.get("export_tariff", 0.0))

self_use_ratio = float(st.session_state.get("self_use", 0.6))  # 0–1

discount_rate = 0.08            # WACC
degradation = 0.005             # 0.5 %/year
om_ratio = 0.015                # 1.5 % of CAPEX / year

inv_replacement_year = 12
inv_replacement_cost = 80_000   # THB

# -------------------------------
# ENERGY MODEL (PV OUTPUT)
# -------------------------------
# ต้องเป็น PV energy ไม่ใช่ load
E_year_1 = E_est_day * 365      # kWh/year (จาก PV sizing)

if E_year_1 <= 0 or CAPEX <= 0:
    st.warning("⚠️ Financial calculation not possible")
    st.stop()

# -------------------------------
# CASHFLOW CALCULATION
# -------------------------------
cashflows = [-CAPEX]
discounted_cum = -CAPEX

simple_payback = None
discounted_payback = None

for y in range(1, project_life + 1):

    # PV degradation
    E_y = E_year_1 * ((1 - degradation) ** (y - 1))

    # Revenue split
    revenue = (
        E_y * self_use_ratio * tariff_self +
        E_y * (1 - self_use_ratio) * tariff_export
    )

    # O&M
    om_cost = CAPEX * om_ratio

    # Inverter replacement
    replacement = inv_replacement_cost if y == inv_replacement_year else 0

    net_cf = revenue - om_cost - replacement
    cashflows.append(net_cf)

    # Payback tracking
    if simple_payback is None:
        if sum(cashflows[1:]) >= CAPEX:
            simple_payback = y

    discounted_cf = net_cf / ((1 + discount_rate) ** y)
    discounted_cum += discounted_cf

    if discounted_payback is None and discounted_cum >= 0:
        discounted_payback = y

# -------------------------------
# FINANCIAL METRICS
# -------------------------------
npv = sum(cf / ((1 + discount_rate) ** i) for i, cf in enumerate(cashflows))
irr_val = irr(cashflows)

st.markdown(
    f"""
###  ผลการวิเคราะห์ทางการเงิน (Financial Results – PVsyst-grade)

**เศรษฐศาสตร์ของระบบ (System Economics)**
- เงินลงทุนเริ่มต้น (CAPEX): **{CAPEX:,.0f} THB**
- พลังงานปีแรก (Year-1 Energy): **{E_year_1:,.0f} kWh/year**
- อัตราการใช้ไฟเอง (Self-consumption): **{self_use_ratio*100:.0f} %**

**ตัวชี้วัดทางการเงิน (Financial Indicators)**
- ระยะเวลาคืนทุนแบบธรรมดา (Simple Payback):  
  **{simple_payback if simple_payback else '>' + str(project_life)} ปี (years)**

- ระยะเวลาคืนทุนแบบคิดลด (Discounted Payback):  
  **{discounted_payback if discounted_payback else '>' + str(project_life)} ปี (years)**

- มูลค่าปัจจุบันสุทธิ (NPV) @ {discount_rate*100:.0f}%:  
  **{npv:,.0f} THB**

- อัตราผลตอบแทนภายใน (IRR):  
  **{irr_val*100:.1f} %**

**หมายเหตุเชิงวิศวกรรม (Engineering Notes)**
- คิดค่าการเสื่อมสภาพของแผง PV (PV degradation) = **0.5 %/year**
- ค่าบำรุงรักษาระบบ (O&M) = **1.5 % ของ CAPEX ต่อปี**
- ค่าทดแทนอินเวอร์เตอร์ (Inverter replacement) ปีที่ **{inv_replacement_year}**
- รายได้แยกการใช้ไฟเอง (Self-use) และไฟส่งออก (Export)
"""
)


def safe_round(value, digits=2):
    try:
        if value is None:
            return "N/A"
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)) and np.isfinite(value):
            return round(float(value), digits)
        return "N/A"
    except:
        return "N/A"


# =========================================================
# AI RESULT STORAGE (Production Safe)
# =========================================================

if "ai_result" not in st.session_state:
    st.session_state["ai_result"] = None

if "ai_loading" not in st.session_state:
    st.session_state["ai_loading"] = False


if st.button("Generate AI Recommendation", disabled=st.session_state["ai_loading"]):

    # -------------------------------------------------
    # Validate prerequisites
    # -------------------------------------------------
    if not st.session_state.get("run_design", False):
        st.warning("⚠️ Please run PV system design first.")
        st.stop()

    panels_df = st.session_state.get("panels_db", pd.DataFrame())
    inverters_df = st.session_state.get("inverters_db", pd.DataFrame())

    if panels_df.empty or inverters_df.empty:
        st.warning("⚠️ Equipment database not loaded.")
        st.stop()

    # -------------------------------------------------
    # Run AI Engine
    # -------------------------------------------------
    st.session_state["ai_loading"] = True

    try:
        with st.spinner("AI selecting optimal equipment..."):

            ai_result = ai_select_from_database(
                panels_df=panels_df,
                inverters_df=inverters_df,
                dc_capacity=dc_capacity,
                dc_ac_ratio=dc_ac_ratio,
                area=area,
                GEMINI_KEY=GEMINI_KEY,
                OPENAI_KEY=OPENAI_KEY
            )

            st.session_state["ai_result"] = ai_result

        st.success("✅ AI recommendation generated successfully.")

    except Exception as e:
        st.session_state["ai_result"] = "AI execution failed."
        st.error(f"❌ AI Error: {str(e)}")

    finally:
        st.session_state["ai_loading"] = False


# -------------------------------------------------
# Display Result (Always Visible After Generation)
# -------------------------------------------------
if st.session_state.get("ai_result"):
    st.markdown("## AI Recommendation Result")
    st.code(st.session_state["ai_result"])






st.header(" Export IEEE Engineering Paper")

if st.button(" Generate IEEE Paper", key="ieee_export_btn"):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="IEEE_Title",
        fontName="TH-B",
        fontSize=18,
        alignment=TA_CENTER,
        spaceAfter=14
    ))

    styles.add(ParagraphStyle(
        name="IEEE_Section",
        fontName="TH-B",
        fontSize=14,
        spaceBefore=12,
        spaceAfter=6
    ))

    styles.add(ParagraphStyle(
        name="IEEE_Body",
        fontName="TH",
        fontSize=12,
        leading=16,
        alignment=TA_JUSTIFY
    ))

    story = []

    # =====================================================
    # TITLE
    # =====================================================
    story.append(Paragraph(
        "Design and Optimization of Rooftop Solar PV System with AI-Assisted Component Selection",
        styles["IEEE_Title"]
    ))

    story.append(Spacer(1, 8))

    # =====================================================
    # ABSTRACT
    # =====================================================
    story.append(Paragraph("Abstract", styles["IEEE_Section"]))

    story.append(Paragraph(
        f"""
This paper presents the engineering design and economic evaluation of a rooftop
solar photovoltaic (PV) system sized at {dc_capacity:.2f} kWp.
The system is designed based on peak sun hours ({H_sun:.2f} h/day),
performance ratio ({PR:.2f}), and rooftop constraints ({area:.1f} m²).
A deterministic calculation approach is applied for system sizing,
while a large language model (LLM) is utilized for database-assisted
component selection. Financial feasibility including IRR and payback
period is evaluated to determine project viability.
""",
        styles["IEEE_Body"]
    ))

    # =====================================================
    # I. INTRODUCTION
    # =====================================================
    story.append(Paragraph("I. INTRODUCTION", styles["IEEE_Section"]))

    story.append(Paragraph(
        """
Rooftop solar photovoltaic systems are increasingly adopted
for residential and commercial applications.
Proper engineering design is essential to ensure electrical safety,
performance optimization, and financial feasibility.
""",
        styles["IEEE_Body"]
    ))

    # =====================================================
    # II. SYSTEM DESIGN METHODOLOGY
    # =====================================================
    story.append(Paragraph("II. SYSTEM DESIGN METHODOLOGY", styles["IEEE_Section"]))

    story.append(Paragraph(
        f"""
The required PV capacity is calculated using the daily energy demand
({E_day:.2f} kWh/day), peak sun hours, and performance ratio.
The DC/AC ratio is maintained at {dc_ac_ratio:.2f} to ensure inverter
loading optimization and clipping control.
""",
        styles["IEEE_Body"]
    ))

    # =====================================================
    # III. ENGINEERING RESULTS
    # =====================================================
    story.append(Paragraph("III. ENGINEERING RESULTS", styles["IEEE_Section"]))

    results_table = Table([
        ["Parameter", "Value"],
        ["PV Capacity (kWp)", f"{dc_capacity:.2f}"],
        ["DC/AC Ratio", f"{dc_ac_ratio:.2f}"],
        ["Panels per String", str(panels_per_string)],
        ["Number of Strings", str(strings_used)],
    ], colWidths=[230, 230])

    results_table.setStyle(TableStyle([
        ("FONT", (0,0), (-1,-1), "TH"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
    ]))

    story.append(results_table)

    # =====================================================
    # IV. AI-ASSISTED COMPONENT SELECTION
    # =====================================================
    story.append(Paragraph("IV. AI-ASSISTED COMPONENT SELECTION", styles["IEEE_Section"]))

    ai_result = st.session_state.get("ai_result", "No AI result available.")

    story.append(Paragraph(
        ai_result.replace("\n", "<br/>"),
        styles["IEEE_Body"]
    ))

    # =====================================================
    # V. FINANCIAL ANALYSIS
    # =====================================================
    story.append(Paragraph("V. FINANCIAL ANALYSIS", styles["IEEE_Section"]))

    story.append(Paragraph(
        f"""
The financial evaluation indicates a simple payback period of
{simple_payback} years and an internal rate of return (IRR)
of {irr_val*100:.2f}%.
""",
        styles["IEEE_Body"]
    ))

    # =====================================================
    # VI. CONCLUSION
    # =====================================================
    story.append(Paragraph("VI. CONCLUSION", styles["IEEE_Section"]))

    story.append(Paragraph(
        """
The designed solar PV system satisfies engineering constraints
and demonstrates economic feasibility.
The integration of deterministic calculation with AI-assisted
database selection enhances engineering workflow efficiency
while maintaining technical reliability.
""",
        styles["IEEE_Body"]
    ))

    # =====================================================
    # BUILD
    # =====================================================
    doc.build(story)

    st.download_button(
        "Download PDF",
        data=buffer.getvalue(),
        file_name="IEEE_Solar_PV_Paper.pdf",
        mime="application/pdf",
        key="download_ieee_btn"
    )



