import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import base64
import os
import zipfile
import gc 
import json
import pickle
from datetime import datetime

# --- 1. ACCESS CONTROL ---
def check_password():
    def password_entered():
        admin_p = st.secrets["ADMIN_PASSWORD"]
        guest_p = st.secrets["GUEST_PASSWORD"]
        entered = st.session_state.get("password_input", "")
        if entered in [admin_p, guest_p]:
            st.session_state["password_correct"] = True
            st.session_state["is_admin"] = (entered == admin_p)
            if "password_input" in st.session_state:
                del st.session_state["password_input"]
        else:
            st.session_state["password_correct"] = False
            
    if "password_correct" not in st.session_state:
        st.markdown("<h2 style='text-align: center; color: #002147;'>BOTMan Betting Systems</h2>", unsafe_allow_html=True)
        st.text_input("Enter Password", type="password", on_change=password_entered, key="password_input")
        return False
    return st.session_state.get("password_correct", False)

if not check_password(): st.stop() 

# --- 2. PAGE CONFIG ---
st.set_page_config(page_title="BOTMan Betting Systems", page_icon="BOTManLogo.png", layout="wide", initial_sidebar_state="expanded")

# --- 3. THE "MORNING REFRESH" ENGINE (SPEED FIX) ---
@st.cache_data(show_spinner="Calculating Morning Data...")
def get_processed_master_data(_df_all, _df_today, _model, _shadow_model, feats, shadow_feats):
    """
    Calculates all ranks and probabilities for the entire dataset upfront.
    This is what makes the Race Analysis screen super fast.
    """
    def process_block(df, is_today=False):
        df = df.copy()
        # ML Probabilities
        df['ML_Prob'] = _model.predict_proba(df[feats].fillna(0))[:, 1]
        df['Shadow_Prob'] = _shadow_model.predict_proba(df[shadow_feats].fillna(0))[:, 1]
        
        # Rankings
        group_cols = ['Time', 'Course'] if is_today else ['Date', 'Time', 'Course']
        df['Rank'] = df.groupby(group_cols)['ML_Prob'].rank(ascending=False, method='min')
        df['Pure Rank'] = df.groupby(group_cols)['Shadow_Prob'].rank(ascending=False, method='min')
        
        # Value Calculations
        df['Value Price'] = 1 / df['ML_Prob']
        
        # No. of Top Mauve Logic
        if 'No. of Top' in df.columns:
            df['No. of Top'] = pd.to_numeric(df['No. of Top'], errors='coerce').fillna(0)
            df['Max_Top'] = df.groupby(group_cols)['No. of Top'].transform('max')
            df['isM'] = (df['No. of Top'] == df['Max_Top']) & (df['No. of Top'] > 0)
        
        return df

    processed_all = process_block(_df_all)
    processed_today = process_block(_df_today, is_today=True) if _df_today is not None else None
    
    return processed_all, processed_today

# --- 4. DATA LOADING ---
@st.cache_resource(show_spinner=False)
def load_base_resources():
    # Load Models and Raw Data (Same as your original logic)
    # [Internal logic remains identical to your provided snippet for model training/loading]
    # Returning placeholders for brevity, use your full training block here:
    return model, feats, shadow_model, shadow_feats, df_hist, df_live, df_today_raw, df_all_raw

# Execute Resources
model, feats, shadow_model, shadow_feats, df_hist, df_live, df_today_raw, df_all_raw = load_base_resources()

# Execute Morning Refresh (Pre-calculates everything)
df_all, df_today = get_processed_master_data(df_all_raw, df_today_raw, model, shadow_model, feats, shadow_feats)

# --- 5. SIDEBAR NAVIGATION ---
with st.sidebar:
    if os.path.exists("BOTManLogo.png"):
        st.image("BOTManLogo.png", use_container_width=True)
    
    st.title("Main Menu")
    app_mode = st.radio("Navigate To:", 
        ["📅 Daily Predictions", "📊 AI Top 2 Results", "🧠 General Systems", "🛠️ System Builder", "🏇 Race Analysis"])
    
    st.markdown("---")
    if st.session_state.get("is_admin"):
        st.subheader("Admin Controls")
        if st.button("⚡ Daily Refresh", use_container_width=True):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()
        
        show_insights = st.checkbox("Show Admin Insights", value=st.session_state.get("show_admin_insights", False))
        st.session_state.show_admin_insights = show_insights

# --- 6. PAGE ROUTING ---
# Based on the Sidebar selection, we render the screens.
# Note: Because the 'df_today' and 'df_all' are already pre-calculated by our Morning Engine,
# the Race Analysis and Predictions will load instantly.

if app_mode == "📅 Daily Predictions":
    # [Insert your Tab 1 Logic here, but use the pre-calculated df_today]
    st.header("📅 Daily Top 2 Predictions")
    # ... (rest of your tab1 code)

elif app_mode == "📊 AI Top 2 Results":
    st.header("📊 AI Performance Dashboard")
    # ... (rest of your tab2 code)

elif app_mode == "🏇 Race Analysis":
    st.header("🏇 Super-Fast Race Analysis")
    # This screen now uses the pre-calculated 'Pure Rank' and 'Value Price'
    # from the Morning Refresh engine.
    if df_today is not None:
        # [Insert your Tab 5 Logic here - it will now be instant!]
        pass
