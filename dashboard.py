import streamlit as st
import sqlite3
import pandas as pd
import os
from PIL import Image

st.set_page_config(page_title="ANPR Dashboard", layout="wide")

DB_PATH = "anpr_results.db"

def get_data():
    conn = sqlite3.connect(DB_PATH)
    try:
        query = "SELECT * FROM detections ORDER BY timestamp DESC"
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        return pd.DataFrame()
    finally:
        conn.close()
    return df

st.title("🚗 Number Plate Recognition Dashboard")

# Refresh Button
if st.button("Refresh Data"):
    st.rerun()

if os.path.exists(DB_PATH):
    df = get_data()
    
    if not df.empty:
        # KPI Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Detections", len(df))
        c2.metric("Unique Vehicles", df['vehicle_id'].nunique() if 'vehicle_id' in df.columns else "N/A")
        
        avg_conf = df['confidence'].mean() if 'confidence' in df.columns else 0.0
        c3.metric("Avg Confidence", f"{avg_conf:.2f}")

        st.markdown("---")

        # Recent Detections Grid
        st.subheader("📋 Recent Detections (Last 10)")
        
        recent_df = df.head(10)
        
        for idx, row in recent_df.iterrows():
            with st.container():
                col1, col2, col3, col4 = st.columns([1, 2, 2, 2])
                
                with col1:
                    img_path = row.get('image_path')
                    if img_path and os.path.exists(img_path):
                        try:
                            image = Image.open(img_path)
                            st.image(image, width=100)
                        except:
                            st.text("Error loading image")
                    else:
                        st.text("No Image")
                
                with col2:
                    st.markdown(f"**Plate:** `{row.get('plate_number', 'Unknown')}`")
                
                with col3:
                    conf = row.get('confidence', 0.0)
                    st.write(f"Confidence: {conf:.2f}")
                    
                with col4:
                    vehicle_id = row.get('vehicle_id', 'N/A')
                    timestamp = row.get('timestamp', '')
                    st.caption(f"ID: {vehicle_id} | Time: {timestamp}")
                
                st.divider()

        # Full Data Table with Search
        st.subheader("🗃️ Complete Log History")
        search_term = st.text_input("Search Plate Number:")
        
        if search_term:
            filtered_df = df[df['plate_number'].astype(str).str.contains(search_term, case=False, na=False)]
            st.dataframe(filtered_df, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)

    else:
        st.info("Database exists but contains no records yet. Run `app.py` to start detecting.")
else:
    st.warning("Database not found! Please run `python app.py` first.")
