import openai
import streamlit as st

# Get the key from secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

try:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "What is the UNSPSC code for laptop computers?"}
        ],
        temperature=0.2
    )
    print("✅ API key works!")
    print("Response:\n", response["choices"][0]["message"]["content"])

except Exception as e:
    print("❌ Something went wrong.")
    print(e)
