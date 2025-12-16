# app.py
# ===== Imports =====
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF
from serpapi import GoogleSearch
import mimetypes
import openai
import pdfplumber
import json
import requests
from bs4 import BeautifulSoup
import streamlit as st
import pytesseract
import re
from PIL import Image, ImageFilter, ImageOps
from io import BytesIO

OPENAI_KEY = "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"
openai.api_key = OPENAI_KEY

SERPAPI_KEY = "d372cb93c1f53bf1a94b225ed594171d04a4663fe7cb176ae8b77272250aff4b"

# ===== การตั้งค่าแอป =====
st.set_page_config(page_title="Solar Rooftop Designer", page_icon="🔆", layout="wide")
st.title(" Solar Rooftop Designer")

# ===== Utilities =====
def irr(cashflows, guess: float = 0.1, max_iter: int = 100, tol: float = 1e-6) -> float:
    """คำนวณ IRR จาก cashflows ด้วย Newton-Raphson"""
    r = guess
    for _ in range(max_iter):
        f = sum(cf / ((1 + r) ** i) for i, cf in enumerate(cashflows))
        df = sum(-i * cf / ((1 + r) ** (i + 1)) for i, cf in enumerate(cashflows))
        if abs(df) < 1e-12: break
        r_new = r - f / df
        if abs(r_new - r) < tol: return r_new
        r = r_new
    return r

def serpapi_search(query: str, location: str = "Thailand", num: int = 10):
    """ค้นหาด้วย SerpAPI"""
    params = {"engine": "google", "q": query, "location": location, "num": num, "api_key": SERPAPI_KEY}
    search = GoogleSearch(params)
    results = search.get_dict()
    return [{"title": item.get("title", ""), "snippet": item.get("snippet", ""), "link": item.get("link", "")}
            for item in results.get("organic_results", [])]


## ===== OCR Extract by Keyword =====
def extract_specs_from_image_by_keyword(uploaded_file, keyword: str) -> dict:

    def preprocess_image(img: Image.Image) -> Image.Image:
        img = img.convert("L")
        img = img.resize((int(img.width * 1.3), int(img.height * 1.3)), Image.LANCZOS)
        img = img.filter(ImageFilter.MedianFilter(size=3))
        img = ImageOps.autocontrast(img)
        return img.point(lambda p: 255 if p > 150 else 0)

    try:
        raw = uploaded_file.read()
        uploaded_file.seek(0)

        img = Image.open(BytesIO(raw))

        # ---- performance: crop เฉพาะส่วนกลาง ----
        w, h = img.size
        img = img.crop((0, int(h * 0.15), w, int(h * 0.65)))

        pre = preprocess_image(img)

        text = pytesseract.image_to_string(
            pre,
            lang="eng",
            config="--psm 6 -c preserve_interword_spaces=1"
        )

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        block = next(
            ("\n".join(lines[i:i+15]) for i, ln in enumerate(lines) if keyword in ln),
            ""
        )

        specs = {
            "Pm":  r"(?:Maximum Power|Pmax).*?(\d{3,4})\s*W[pP]",
            "Vmp": r"(?:Power Voltage|Vmp).*?(\d{2,3}\.\d{2})\s*V",
            "Imp": r"(?:Power Current|Imp).*?(\d{2,3}\.\d{2})\s*A",
            "Voc": r"(?:Open[- ]?circuit Voltage|Voc).*?(\d{2,3}\.\d{2})\s*V",
            "Isc": r"(?:Short[- ]?circuit Current|Isc).*?(\d{2,3}\.\d{2})\s*A",
        }

        result = {}
        for k, p in specs.items():
            m = re.search(p, block)
            if m:
                try:
                    result[k] = float(m.group(1))
                except Exception:
                    pass

        return result

    except Exception as e:
        print("OCR spec extract error:", e)
        return {}


# ===== Auto Extract Text =====
def auto_extract_text(source: str, timeout: int = 12) -> str:

    text = ""
    try:
        is_url = source.startswith("http")
        mime, _ = mimetypes.guess_type(source)

        if is_url:
            resp = requests.get(source, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                return ""
            data = resp.content
        else:
            data = None

        if mime and "pdf" in mime:
            pdf_src = BytesIO(data) if is_url else source
            with pdfplumber.open(pdf_src) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

        elif mime and "image" in mime:
            img = Image.open(BytesIO(data)) if is_url else Image.open(source)
            pre = extract_specs_from_image_by_keyword(img, "")
            text = pytesseract.image_to_string(pre, lang="eng")

        else:
            if is_url:
                soup = BeautifulSoup(resp.text, "html.parser")
                text = soup.get_text(separator="\n")
            else:
                with open(source, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

    except Exception as e:
        print("auto_extract_text error:", e)

    return text.strip()


# ===== Fetch Page Text =====
def fetch_page_text(url: str, timeout: int = 12, max_chars: int = 30000) -> str:

    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return ""

        if "html" in resp.headers.get("Content-Type", "").lower() or resp.text.lstrip().startswith("<"):
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "noscript", "header", "footer", "svg"]):
                tag.decompose()
            text = re.sub(r"\s+", " ", soup.get_text(separator="\n")).strip()
            return text[:max_chars]

        return ""
    except Exception:
        return ""


# ===== OpenAI Extract Specs =====
def openai_extract_specs(datasheet_text: str, kind: str = "module", max_input_chars: int = 20000) -> dict:

    if not datasheet_text or not datasheet_text.strip():
        return {}

    text = datasheet_text.strip()[:max_input_chars]

    user_prompt = (
        "คุณจะได้รับข้อความจาก data sheet ของแผงโซลาร์เซลล์ กรุณาดึงค่าต่อไปนี้และตอบกลับเป็น JSON เท่านั้น:\n"
        "- Vmp (V)\n- Voc (V)\n- Imp (A)\n- Isc (A)\n- Pm (W)\n\n"
        "ถ้าไม่พบค่า ให้ใส่ null สำหรับค่านั้น\n\n"
        f"ข้อความ:\n{text}"
    ) if kind == "module" else (
        "คุณจะได้รับข้อความจาก data sheet ของอินเวอร์เตอร์ กรุณาดึงค่าต่อไปนี้และตอบกลับเป็น JSON เท่านั้น:\n"
        "- inv_power_ac (W)\n- inv_v_dc_max (V)\n- inv_i_sc_max (A)\n- inv_pv_power_max (W)\n\n"
        "ถ้าไม่พบค่า ให้ใส่ null สำหรับค่านั้น\n\n"
        f"ข้อความ:\n{text}"
    )

    try:
        chat = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "คุณเป็นตัวดึงข้อมูลเชิงโครงสร้างจากเอกสาร ให้ตอบกลับเป็น JSON เท่านั้น"},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            timeout=30
        )

        content = re.sub(r"```json|```", "", chat.choices[0].message["content"]).strip()

        try:
            return json.loads(content)
        except Exception:
            m = re.search(r"\{.*\}", content, flags=re.S)
            return json.loads(m.group(0)) if m else {}

    except Exception:
        return {}


# ===== Safe Float =====
def safe_float(x, default=None):

    try:
        if x is None:
            return default
        s = re.sub(r"[,\s%A-Za-zก-ฮ\u0E00-\u0E7F]", "", str(x).strip())
        s = re.sub(r"[^\d\.\-eE]", "", s)
        return float(s) if s not in ("", ".", "-", "nan") else default
    except Exception:
        return default



# ===== Session state defaults =====
defaults_module = {"Vmp": 43.71, "Voc": 54.08, "Imp": 14.30, "Isc": 15.03, "Pm": 625.0}
defaults_inverter = {"inv_power_ac": 10000.0, "inv_v_dc_max": 600.0, "inv_i_sc_max": 20.0, "inv_pv_power_max": 15000.0}
for k, v in defaults_module.items():
    st.session_state.setdefault(k, v)
for k, v in defaults_inverter.items():
    st.session_state.setdefault(k, v)



# ===== Sidebar inputs (จะถูกแทนค่าหลังดึง data sheet) =====
st.sidebar.header("ข้อมูลโหลดและสภาพแวดล้อม")
E_day = st.sidebar.number_input("พลังงานที่ใช้ช่วงกลางวัน (kWh/day)", min_value=0.0, value=30.0, step=0.5)
H_sun = st.sidebar.number_input("ชั่วโมงแสงอาทิตย์เฉลี่ยต่อวัน (hr)", min_value=0.0, value=4.5, step=0.1)
PR = st.sidebar.slider("Performance Ratio (PR)", 0.6, 0.9, 0.80, 0.01)

st.sidebar.header("ข้อมูลแผงโซลาร์เซลล์ (จาก data sheet)")
Vmp = st.sidebar.number_input("Vmp (V)", value=st.session_state["Vmp"])
Voc = st.sidebar.number_input("Voc (V)", value=st.session_state["Voc"])
Imp = st.sidebar.number_input("Imp (A)", value=st.session_state["Imp"])
Isc = st.sidebar.number_input("Isc (A)", value=st.session_state["Isc"])
Pm = st.sidebar.number_input("กำลังสูงสุดต่อแผง (W)", value=st.session_state["Pm"])

st.sidebar.header("ข้อมูลอินเวอร์เตอร์ (จาก data sheet)")
inv_power_ac = st.sidebar.number_input("กำลังอินเวอร์เตอร์ AC (W)", value=st.session_state["inv_power_ac"])
inv_v_dc_max = st.sidebar.number_input("DC max voltage (V)", value=st.session_state["inv_v_dc_max"])
inv_i_sc_max = st.sidebar.number_input("Max. short-circuit current (A)", value=st.session_state["inv_i_sc_max"])
inv_pv_power_max = st.sidebar.number_input("Max. PV power (W)", value=st.session_state["inv_pv_power_max"])

st.sidebar.header("เศรษฐศาสตร์")
CAPEX = st.sidebar.number_input("ราคาติดตั้ง (บาท)", value=350000.0, step=1000.0)
tariff = st.sidebar.number_input("ค่าไฟฟ้า (บาท/kWh)", value=4.2, step=0.1)
years = st.sidebar.number_input("ระยะเวลาวิเคราะห์ (ปี)", value=20, step=1)
discount_rate = st.sidebar.slider("Discount Rate (ต่อปี)", 0.0, 0.2, 0.08, 0.01)
degradation = st.sidebar.slider("การเสื่อมสมรรถนะ (% ต่อปี)", 0.0, 2.0, 0.7, 0.1)

# ===== ปุ่มวิเคราะห์ =====
if st.sidebar.button("วิเคราะห์"):
    # อัปเดตค่าใน session_state
    st.session_state["E_day"] = E_day
    st.session_state["H_sun"] = H_sun
    st.session_state["PR"] = PR
    st.session_state["Vmp"] = Vmp
    st.session_state["Voc"] = Voc
    st.session_state["Imp"] = Imp
    st.session_state["Isc"] = Isc
    st.session_state["Pm"] = Pm
    st.session_state["inv_power_ac"] = inv_power_ac
    st.session_state["inv_v_dc_max"] = inv_v_dc_max
    st.session_state["inv_i_sc_max"] = inv_i_sc_max
    st.session_state["inv_pv_power_max"] = inv_pv_power_max
    st.session_state["CAPEX"] = CAPEX
    st.session_state["tariff"] = tariff
    st.session_state["years"] = years
    st.session_state["discount_rate"] = discount_rate
    st.session_state["degradation"] = degradation

    st.success("อัปเดตข้อมูลทั้งหมดเรียบร้อยแล้ว")





# ===== Section: ค้นหาและดึงสเปคจาก data sheet =====

#

st.markdown(" ค้นหาสเปคจาก data sheet ")

colA, colB = st.columns(2)

# ===== ฝั่งแผงโซลาร์ =====
with colA:
    st.subheader("แผงโซลาร์ (PV Module)")
    module_query = st.text_input("คำค้นหาแผง", value="", key="module_query")

    if "module_results" not in st.session_state:
        st.session_state["module_results"] = []

    if st.button("ค้นหาแผง", key="search_module"):
        st.session_state["module_results"] = serpapi_search(module_query)

    if st.session_state["module_results"]:
        options = [f"{i+1}. {r['title']}" for i, r in enumerate(st.session_state["module_results"])]
        idx = st.selectbox("เลือกผลลัพธ์", list(range(len(options))),
                           format_func=lambda i: options[i], key="module_select")
        selected = st.session_state["module_results"][idx]
        st.write(f"ลิงก์ที่เลือก: {selected['link']}")

    # ช่องอัปโหลดไฟล์ datasheet
    uploaded_module = st.file_uploader("อัปโหลดไฟล์ datasheet ของแผง ",
                                       type=["pdf", "png", "jpg", "jpeg"], key="upload_module")

    if uploaded_module is not None and st.button("อัปเดตสเปคแผง", key="update_module"):
        text = auto_extract_text(uploaded_module)
        specs = openai_extract_specs(text, kind="module") if text else {}
        if specs:
            st.json(specs)
            for k in ["Vmp", "Voc", "Imp", "Isc", "Pm"]:
                st.session_state[k] = safe_float(specs.get(k), st.session_state[k])
            st.success("อัปเดตค่าสเปคแผงใน Sidebar แล้ว")
        else:
            st.error("ไม่สามารถสกัดค่าสเปคจากไฟล์ที่อัปโหลดได้")

# ===== ฝั่งอินเวอร์เตอร์ =====
with colB:
    st.subheader("อินเวอร์เตอร์ (Inverter)")
    inverter_query = st.text_input("คำค้นหาอินเวอร์เตอร์", value="", key="inverter_query")

    if "inverter_results" not in st.session_state:
        st.session_state["inverter_results"] = []

    if st.button("ค้นหาอินเวอร์เตอร์", key="search_inverter"):
        st.session_state["inverter_results"] = serpapi_search(inverter_query)

    if st.session_state["inverter_results"]:
        options = [f"{i+1}. {r['title']}" for i, r in enumerate(st.session_state["inverter_results"])]
        idx2 = st.selectbox("เลือกผลลัพธ์", list(range(len(options))),
                            format_func=lambda i: options[i], key="inv_select")
        selected2 = st.session_state["inverter_results"][idx2]
        st.write(f"ลิงก์ที่เลือก: {selected2['link']}")

    # ช่องอัปโหลดไฟล์ datasheet
    uploaded_inverter = st.file_uploader("อัปโหลดไฟล์ datasheet ของอินเวอร์เตอร์ (PDF/รูปภาพ)",
                                         type=["pdf", "png", "jpg", "jpeg"], key="upload_inverter")

    if uploaded_inverter is not None and st.button("อัปเดตสเปคอินเวอร์เตอร์", key="update_inverter"):
        text2 = auto_extract_text(uploaded_inverter)
        specs2 = openai_extract_specs(text2, kind="inverter") if text2 else {}
        if specs2:
            st.json(specs2)
            for k in ["inv_power_ac", "inv_v_dc_max", "inv_i_sc_max", "inv_pv_power_max"]:
                st.session_state[k] = safe_float(specs2.get(k), st.session_state[k])
            st.success("อัปเดตค่าสเปคอินเวอร์เตอร์ใน Sidebar แล้ว")
        else:
            st.error("ไม่สามารถสกัดค่าสเปคจากไฟล์ที่อัปโหลดได้")

st.caption("ค้นหา datasheet → เปิดลิงก์ → ดาวน์โหลด/Captureภาพ → อัปโหลดไฟล์ PDF หรือรูปภาพเพื่ออัปเดตค่า")





# ===== Section: คำนวณระบบและตรวจสอบ =====
st.markdown("---")
st.markdown("## คำนวณระบบและตรวจสอบการจัด String/อินเวอร์เตอร์")

# กำลังติดตั้งที่เหมาะสม
P_pv_kWp = E_day / (H_sun * PR) if H_sun * PR > 0 else 0.0
E_daily_est = P_pv_kWp * H_sun * PR

col1, col2 = st.columns(2)
with col1:
    st.write(f"• กำลังติดตั้งที่เหมาะสม: {P_pv_kWp:.2f} kWp")
with col2:
    st.write(f"• พลังงานผลิตโดยประมาณ: {E_daily_est:.2f} kWh/day")

# ตรวจสอบ String แบบพื้นฐาน
st.subheader("การจัด String")
M_series = st.number_input("จำนวนแผงต่อ String (อนุกรม)", min_value=1, value=10, step=1)
Voc_array = Voc * M_series
Vmp_array = Vmp * M_series
Isc_array = Isc
P_array = Pm * M_series

col3, col4, col5, col6 = st.columns(4)
with col3:
    st.write(f"Voc รวม: {Voc_array:.2f} V")
with col4:
    st.write(f"Vmp รวม: {Vmp_array:.2f} V")
with col5:
    st.write(f"Isc String: {Isc_array:.2f} A")
with col6:
    st.write(f"กำลังรวมต่อ String: {P_array:.2f} W")

st.subheader("ผลตรวจสอบอินเวอร์เตอร์")
checks = [
    ("Voc รวมไม่เกิน DC max", Voc_array <= inv_v_dc_max),
    ("Isc String ไม่เกิน Isc max/MPPT", Isc_array <= inv_i_sc_max),
    ("กำลังรวมไม่เกิน Max PV Power", P_array <= inv_pv_power_max),
]
for name, ok in checks:
    st.write(f"{'✅' if ok else '❌'} {name}")

# ===== 3) Economic Analysis =====
st.header("การวิเคราะห์เศรษฐศาสตร์")
E_year = E_daily_est * 365.0
cashflows = [-CAPEX]
for y in range(1, int(years) + 1):
    factor = (1 - degradation / 100.0) ** (y - 1)
    savings = E_year * tariff * factor
    cashflows.append(savings)

payback = CAPEX / (E_year * tariff) if (E_year * tariff) > 0 else float("inf")
npv_val = sum(cf / ((1 + discount_rate) ** i) for i, cf in enumerate(cashflows))

# IRR calculation
def irr(cashflows, guess: float = 0.1, max_iter: int = 100, tol: float = 1e-6) -> float:
    r = guess
    for _ in range(max_iter):
        f = sum(cf / ((1 + r) ** i) for i, cf in enumerate(cashflows))
        df = sum(-i * cf / ((1 + r) ** (i + 1)) for i, cf in enumerate(cashflows))
        if abs(df) < 1e-12:
            break
        r_new = r - f / df
        if abs(r_new - r) < tol:
            return r_new
        r = r_new
    return r

irr_val = irr(cashflows, guess=0.1)

st.write(f"• ระยะเวลาคืนทุน (Payback Period): {payback:.2f} ปี")
st.write(f"• มูลค่าปัจจุบันสุทธิ (NPV): {npv_val:,.0f} บาท")
st.write(f"• อัตราผลตอบแทนภายใน (IRR): {irr_val*100:.2f}%")

# ===== 4) Export Results =====
st.header("การส่งออกผลลัพธ์")

# Excel Export (English)
df_cf = pd.DataFrame({
    "Year": list(range(0, int(years) + 1)),
    "Cashflow (Baht)": cashflows
})
excel_buffer = BytesIO()
with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
    df_cf.to_excel(writer, sheet_name="Cashflows", index=False)
excel_buffer.seek(0)

st.download_button(
    label=" Download Excel (Cashflows)",
    data=excel_buffer,
    file_name="solar_economics.xlsx"
)

# ===== สร้างกราฟสำหรับ PDF =====
years_list = list(range(1, int(years) + 1))
production = [E_daily_est * 365 * ((1 - degradation/100) ** (y-1)) for y in years_list]

years_with_0 = [0] + years_list
yearly_savings = [E_daily_est * 365 * tariff * ((1 - degradation/100) ** (y-1)) for y in years_list]
cashflows = [-CAPEX] + yearly_savings
cumulative_cf = np.cumsum(cashflows)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ===== สร้างกราฟคู่สำหรับ PDF =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# กราฟซ้าย: Energy Production
ax1 = axes[0]
ax1.plot(years_list, production, marker='o', color='green', label="Yearly Energy Production")
ax1.set_xlabel("Year")
ax1.set_ylabel("Energy Production (kWh/year)")
ax1.set_title("Yearly Electricity Production with Degradation")
ax1.grid(True, linestyle=':', alpha=0.4)
ax1.legend()

# กราฟขวา: Cashflow
ax2 = axes[1]
ax2.bar(years_with_0, cashflows, color=['crimson'] + ['steelblue']*len(yearly_savings), label="Yearly Cashflow")
ax2.plot(years_with_0, cumulative_cf, marker='o', color='orange', label="Cumulative Cashflow")
ax2.axhline(0, color='red', linestyle='--', label="Break-even Line")
ax2.set_xlabel("Year")
ax2.set_ylabel("Cashflow (Baht)")
ax2.set_title("Yearly and Cumulative Cashflows")
ax2.grid(True, linestyle=':', alpha=0.4)
ax2.legend()

# บันทึกกราฟเป็นรูปภาพ
fig.savefig("graphs.png")
plt.close(fig)

# ===== สร้าง PDF =====
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

# รายงานตัวเลข
pdf.cell(0, 10, txt="Solar Rooftop Economic Analysis Report", ln=True)
pdf.ln(5)
pdf.cell(0, 10, txt=f"Optimal Installed Capacity: {P_pv_kWp:.2f} kWp", ln=True)
pdf.cell(0, 10, txt=f"Estimated Daily Energy Production: {E_daily_est:.2f} kWh/day", ln=True)
pdf.ln(5)
pdf.cell(0, 10, txt=f"Payback Period: {payback:.2f} years", ln=True)
pdf.cell(0, 10, txt=f"Net Present Value (NPV): {npv_val:,.0f} Baht", ln=True)
pdf.cell(0, 10, txt=f"Internal Rate of Return (IRR): {irr_val*100:.2f}%", ln=True)

# แทรกรูปกราฟคู่
pdf.add_page()
pdf.image("graphs.png", x=10, y=30, w=180)

# แปลงเป็น BytesIO สำหรับดาวน์โหลด
pdf_bytes = pdf.output(dest='S').encode('latin1')
pdf_buffer = BytesIO(pdf_bytes)

# ปุ่มดาวน์โหลด PDF
st.download_button(
    label=" Download PDF (Summary + Graphs)",
    data=pdf_buffer,
    file_name="solar_summary_with_graphs.pdf")

