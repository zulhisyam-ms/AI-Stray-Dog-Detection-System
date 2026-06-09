#The production frontend code designed to dynamically ingest log events, draw analytical visualizations, and stream network media blocks

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import os
import base64
from PIL import Image

st.set_page_config(page_title="Stray Dog Detection Dashboard", layout="wide")

# Force Premium Corporate White Layout via custom CSS inject
st.markdown("""
<style>
.stApp { background-color: #ffffff; color: #2c3e50; }
[data-testid="stDataFrame"] { background-color: #ffffff !important; box-shadow: 0 2px 6px rgba(0,0,0,0.05); }
h1, h2, h3 { color: #2c3e50 !important; font-family: 'Segoe UI', sans-serif; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🐕 Smart Stray Dog Monitoring & Analytics Dashboard")

video_file_path = "encoded_video.mp4"
col1, col2 = st.columns([2, 1])

with col1:
    if os.path.exists(video_file_path):
        st.markdown("### 📹 Live/Annotated Video Feed Preview")
        with open(video_file_path, "rb") as video_file:
            encoded_video = base64.b64encode(video_file.read()).decode()
        st.markdown(f'<video width="100%" controls><source src="data:video/mp4;base64,{encoded_video}" type="video/mp4"></video>', unsafe_allow_html=True)

with col2:
    st.markdown("### 📝 System Operation Instructions")
    st.markdown("- **Real-Time Bounding Boxes:** Highlights detected instances with precision localization markers.\n- **Automated Logging:** Aggregates edge timestamps directly to table metrics.\n- **Risk Profiling:** Prioritizes localized zones that show excessive anomaly density.")

# Ingest and display real-time telemetry tables and analytical donut plots
csv_path = "detection_log.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    c1, c2 = st.columns([3, 2])
    
    with c1:
        st.markdown("### 📋 Detection Event Log Sequence")
        st.dataframe(df, use_container_width=True)
        
    with c2:
        st.markdown("### 📍 Geographic Incident Share")
        zone_counts = df["Zone"].value_counts()
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.pie(zone_counts, labels=zone_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
        ax.add_artist(plt.Circle((0,0),0.70,fc='white')) # Donut styling
        st.pyplot(fig)
