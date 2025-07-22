
import streamlit as st
import requests
import json
import pandas as pd
import altair as alt
from datetime import datetime

st.title("English Speaking Fluency Tester")

# Language selector
languages = {
    "English": "en",
    # "Mandarin Chinese": "zh",
    # "Hindi": "hi",
    # "Spanish": "es",
    # "French": "fr",
    # "Arabic": "ar",
    # "Bengali": "bn",
    # "Russian": "ru",
    # "Portuguese": "pt",
    # "Urdu": "ur"
}
selected_language = st.selectbox("Select Language", list(languages.keys()))
language_code = languages[selected_language]

# Audio input (file upload only)
uploaded_file = st.file_uploader(f"Upload Audio in {selected_language}", type=["wav", "mp3", "mpeg"])

if uploaded_file:
    # Send to backend
    files = {"file": ("audio.wav", uploaded_file, "audio/mpeg" if uploaded_file.name.endswith(".mpeg") else "audio/wav")}
    data = {"language": language_code}
    response = requests.post("http://localhost:8000/evaluate", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        
        # Display scores
        st.header("Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Band", result["band_score"])
        col2.metric("Fluency", result["fluency"])
        col3.metric("Grammar", result["grammar"])
        col4.metric("Vocabulary", result["vocabulary"])
        
        # Display transcript
        st.header("Transcript")
        st.write(result["transcript"])
        
        # Display feedback
        st.header("Detailed Feedback")
        for fb in result["feedback"]:
            with st.expander(f"{fb['area'].capitalize()} Issue at {fb['time']}"):
                st.write(f"Issue: {fb['issue']}")
                st.write(f"Suggestion: {fb['suggestion']}")
        
        # Display feedback table
        st.header("All Feedback")
        df_feedback = pd.DataFrame(result["feedback"])
        st.table(df_feedback)
        
        # Display progress chart
        st.header("Progress Over Time")
        with open("../data/history.json", "r") as f:
            history = json.load(f)
        
        df_history = pd.DataFrame(history)
        df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
        df_history = df_history[df_history['language'] == language_code]
        chart = alt.Chart(df_history).mark_line().encode(
            x=alt.X('timestamp:T', title='Date'),
            y=alt.Y('band_score:Q', title='Band Score'),
            tooltip=['timestamp', 'band_score', 'language']
        ).properties(
            width=600,
            height=400
        )
        st.altair_chart(chart, use_container_width=True)








