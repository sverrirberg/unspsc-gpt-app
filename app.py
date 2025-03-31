import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
from typing import Tuple
from openai import OpenAI
import re
import os
from io import BytesIO

# API keys from secrets
gpt_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="UNSPSC Classifier", layout="wide")
st.title("🔍 Greind - UNSPSC via GPT")

st.markdown(
    "Upload a CSV with procurement descriptions, and we’ll classify them using GPT. "
    "We’ll extract the FAMILY-level UNSPSC code from GPT and match it to your approved subset."
)

ALLOWED_SEGMENTS = {"10", "11", "13", "14", "24", "30", "31", "43", "50", "52"}

subset_path = os.path.join("data", "cleaned_unspsc_subset.csv")
if not os.path.exists(subset_path):
    st.error("UNSPSC subset file not found in /data directory.")
    st.stop()
subset_df = pd.read_csv(subset_path, dtype=str)
subset_df.columns = [c.strip().lower() for c in subset_df.columns]
subset_df["code"] = subset_df["code"].str.replace("-", "").str.strip()
subset_df["code"] = subset_df["code"].apply(lambda x: re.sub(r"\.0+$", "", x).zfill(4) if x.isdigit() else x)
subset_df["level"] = subset_df["level"].str.strip().str.lower()

desc_file = st.file_uploader("📄 Upload procurement descriptions CSV", type="csv")
output_format = st.selectbox("📄 Choose output format", ["CSV", "Excel (XLSX)"])

def normalize_code(code: str) -> str:
    code = str(code).strip().replace("-", "")
    code = re.sub(r"\.0+$", "", code)
    return code.zfill(4) if len(code) <= 4 and code.isdigit() else code

def match_family_code_from_gpt_column(gpt_family_code: str) -> Tuple[str, str, str, str]:
    family_code = normalize_code(gpt_family_code)
    match = subset_df[(subset_df["code"] == family_code) & (subset_df["level"] == "family")]
    if not match.empty:
        row = match.iloc[0]
        return row["code"], row["level"], row["subcategory"], row["description"]
    return "", "", "", ""

def gpt_translate_and_classify(desc: str) -> Tuple[str, str, str, str]:
    prompt = f"""
You are a procurement assistant. Given the following product or service description:
"{desc}"

1. Translate it to English (if needed)
2. Suggest the most appropriate 8-digit UNSPSC code
3. Include the 4-digit FAMILY-level UNSPSC code that the 8-digit code belongs to

Only use codes from these segments: 10, 11, 13, 14, 24, 30, 31, 43, 50, 52.

Return in this format:
Translated: <translated_description>
Code: <8-digit UNSPSC code>
Family: <4-digit UNSPSC family code>
Label: <label for item>
"""
    try:
        response = gpt_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        content = response.choices[0].message.content

        translated = code = family = label = ""
        for line in content.splitlines():
            if line.lower().startswith("translated:"):
                translated = line.split(":", 1)[1].strip()
            elif line.lower().startswith("code:"):
                code = line.split(":", 1)[1].strip()
            elif line.lower().startswith("family:"):
                family = line.split(":", 1)[1].strip()
            elif line.lower().startswith("label:"):
                label = line.split(":", 1)[1].strip()

        code = normalize_code(code)
        family = normalize_code(family)

        if code[:2] not in ALLOWED_SEGMENTS:
            return translated, code, family, label

        return translated, code, family, label

    except Exception as e:
        st.warning(f"OpenAI API error: {e}")
        return "", "", "", ""

if desc_file:
    df = pd.read_csv(desc_file)

    if "procurement_description" not in df.columns:
        st.error("CSV must include 'procurement_description'.")
    else:
        if st.button("🚀 Start Classification"):
            results = []
            unmatched = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, row in df.iterrows():
                desc = row["procurement_description"]
                translated, gpt_code, gpt_family, gpt_label = gpt_translate_and_classify(desc)
                matched_code, matched_level, matched_subcategory, matched_desc = match_family_code_from_gpt_column(gpt_family)

                results.append({
                    "original_description": desc,
                    "translated_description": translated,
                    "gpt_unspsc_code": gpt_code,
                    "gpt_unspsc_family": gpt_family,
                    "gpt_unspsc_label": gpt_label,
                    "matched_subset_code": matched_code,
                    "matched_level": matched_level,
                    "matched_subcategory": matched_subcategory,
                    "matched_description": matched_desc,
                })

                if not matched_code:
                    unmatched.append(results[-1])

                percent = int((i + 1) / len(df) * 100)
                progress_bar.progress((i + 1) / len(df))
                status_text.text(f"Processing row {i+1} of {len(df)} ({percent}%)")
                time.sleep(1.0)

            progress_bar.empty()
            status_text.empty()

            result_df = pd.DataFrame(results)
            unmatched_df = pd.DataFrame(unmatched)

            matched = len(result_df) - len(unmatched_df)
            st.write(f"✅ Matched {matched} of {len(result_df)} rows ({round(matched / len(result_df) * 100, 1)}%)")

            fig, ax = plt.subplots(figsize=(1.2, 1.2))
            ax.pie([matched, len(unmatched_df)], labels=["Matched", "Unmatched"], autopct="%1.1f%%", startangle=90, textprops={'fontsize': 6})
            ax.axis("equal")
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
            st.pyplot(fig)

            if output_format == "CSV":
                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download All Results (CSV)", csv, "unspsc_results.csv", "text/csv")
            else:
                try:
                    import xlsxwriter
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        result_df.to_excel(writer, index=False, sheet_name="Results")
                    st.download_button("⬇️ Download All Results (Excel)", output.getvalue(), "unspsc_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                except ImportError:
                    st.error("To export Excel files, please install the 'xlsxwriter' package: pip install xlsxwriter")
