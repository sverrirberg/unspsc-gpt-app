import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
from typing import Tuple
from openai import OpenAI
import re
import os
import tempfile
import glob

# Setup
st.set_page_config(page_title="🔍 Greind - UNSPSC via GPT", layout="wide")
st.title("🔍 Greind - UNSPSC via GPT")

st.markdown(
    "Upload a CSV with procurement descriptions, and we’ll classify them using GPT in batches. "
    "The model will extract a FAMILY-level UNSPSC code, match it against your approved subset, "
    "and present the results in both CSV and Excel formats."
)

# Load OpenAI key
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Load static subset
subset_path = os.path.join("data", "cleaned_unspsc_subset.csv")
subset_df = pd.read_csv(subset_path, dtype=str)
subset_df.columns = [c.strip().lower() for c in subset_df.columns]
subset_df["code"] = subset_df["code"].str.replace("-", "").str.strip().str.zfill(4)
subset_df["level"] = subset_df["level"].str.strip().str.lower()

# File upload
desc_file = st.file_uploader("📄 Upload procurement descriptions CSV", type="csv")

# Helper functions
def normalize_code(code: str) -> str:
    code = str(code).strip().replace("-", "")
    code = re.sub(r"\.0+$", "", code)  # remove float-style .0 endings
    return code.zfill(4) if len(code) <= 4 and code.isdigit() else code

def gpt_translate_and_classify(desc: str) -> Tuple[str, str, str, str]:
    prompt = f"""
You are a specialized procurement classification expert. Your task is to match procurement item descriptions to the most accurate UNSPSC (United Nations Standard Products and Services Code) code:
"{desc}"

1. Translate it to English (if needed)
2. Analyze the procurement description provided
3. Identify the specific product or service category
4. Match it to the most appropriate UNSPSC code (8-digit format)
5. Return ONLY the UNSPSC code and its description in the specified format
6. If you cannot find an exact match, identify the closest appropriate category
7. If you're uncertain about the match, indicate your confidence level
8. Do NOT create fictional UNSPSC codes
9. Suggest the most appropriate 8-digit UNSPSC code

Return in this format:
Translated: <translated_description>
Code: <8-digit UNSPSC code>
Family: <4-digit UNSPSC family code>
Label: <label for item>
"""
    try:
        response = client.chat.completions.create(
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

        return translated, normalize_code(code), normalize_code(family), label
    except Exception as e:
        return "", "", "", f"GPT error: {e}"

def match_family_code(family_code: str) -> Tuple[str, str, str, str]:
    match = subset_df[(subset_df["code"] == family_code) & (subset_df["level"] == "family")]
    if not match.empty:
        row = match.iloc[0]
        return row["code"], row["level"], row["subcategory"], row["description"]
    return "", "", "", ""

# Main classification
if desc_file and "run_classification" not in st.session_state:
    df = pd.read_csv(desc_file)

    if "procurement_description" not in df.columns:
        st.error("CSV must include a column named 'procurement_description'.")
    else:
        if st.button("🚀 Start Classification"):
            st.session_state.run_classification = True
            st.rerun()

if "run_classification" in st.session_state:
    df = pd.read_csv(desc_file)
    CHUNK_SIZE = 500
    total_chunks = (len(df) - 1) // CHUNK_SIZE + 1

    st.info(f"Processing {len(df)} rows in {total_chunks} chunks of {CHUNK_SIZE}...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    timer_text = st.empty()
    start_time = time.time()

    tmp_dir = tempfile.mkdtemp()
    all_chunk_paths = []

    for chunk_index in range(total_chunks):
        chunk_df = df.iloc[chunk_index * CHUNK_SIZE : (chunk_index + 1) * CHUNK_SIZE]
        chunk_results = []

        for i, row in chunk_df.iterrows():
            desc = row["procurement_description"]
            translated, gpt_code, gpt_family, gpt_label = gpt_translate_and_classify(desc)
            matched_code, matched_level, matched_subcategory, matched_desc = match_family_code(gpt_family)

            chunk_results.append({
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

            processed = chunk_index * CHUNK_SIZE + (i % CHUNK_SIZE) + 1
            percent = int(processed / len(df) * 100)
            progress_bar.progress(min(processed / len(df), 1.0))
            elapsed = int(time.time() - start_time)
            timer_text.text(f"⏱️ Time elapsed: {elapsed} seconds")
            status_text.text(f"Processed {processed} of {len(df)} rows ({percent}%)")

        chunk_path = os.path.join(tmp_dir, f"chunk_{chunk_index}.csv")
        pd.DataFrame(chunk_results).to_csv(chunk_path, index=False)
        all_chunk_paths.append(chunk_path)

        time.sleep(1)  # Pause between chunks

    result_df = pd.concat([pd.read_csv(path) for path in all_chunk_paths], ignore_index=True)
    unmatched_df = result_df[result_df["matched_subset_code"] == ""]
    matched = len(result_df) - len(unmatched_df)

    progress_bar.empty()
    status_text.empty()
    timer_text.empty()

    st.success(f"✅ Matched {matched} of {len(result_df)} rows ({round(matched / len(result_df) * 100, 1)}%)")

    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    ax.pie([matched, len(unmatched_df)], labels=["Matched", "Unmatched"], autopct="%1.1f%%", startangle=90, textprops={'fontsize': 6})
    ax.axis("equal")
    st.pyplot(fig)

    csv_data = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Results (CSV)", csv_data, "unspsc_results.csv", "text/csv")

    excel_output = os.path.join(tmp_dir, "unspsc_results.xlsx")
    with pd.ExcelWriter(excel_output, engine="xlsxwriter") as writer:
        result_df.to_excel(writer, index=False, sheet_name="Results")

    with open(excel_output, "rb") as f:
        st.download_button("⬇️ Download Results (Excel)", f.read(), file_name="unspsc_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Clear session to prevent rerun
    del st.session_state["run_classification"]
