# frontend.py
import streamlit as st
import requests

API_URL = st.text_input("Backend URL", "http://localhost:8000/transcribe")

st.title("Meeting Notes & Action Item Extractor")

uploaded = st.file_uploader("Upload meeting audio", type=["mp3","wav","m4a","mp4"])
language = st.text_input("Language (optional, e.g., 'en')", "")

if uploaded and st.button("Transcribe & Extract"):
    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
    params = {}
    if language:
        params["language"] = language
    with st.spinner("Uploading and processing..."):
        resp = requests.post(API_URL, files=files, params=params, timeout=300)
    if resp.status_code != 200:
        st.error(f"Error: {resp.status_code} - {resp.text}")
    else:
        data = resp.json()
        st.subheader("Transcript")
        st.text_area("Transcript", value=data.get("transcript",""), height=200)

        st.subheader("Structured Notes (JSON)")
        st.json(data.get("structured_notes", {}))

        # show action items as table if present
        ais = data.get("structured_notes", {}).get("action_items", [])
        if ais:
            st.subheader("Action Items")
            import pandas as pd
            df = pd.DataFrame(ais)
            st.dataframe(df)
