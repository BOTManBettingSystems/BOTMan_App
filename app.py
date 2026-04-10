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
import io
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

# --- 3. DATA ENGINE (MODIFIED FOR SPEED) ---
@st.cache_resource(show_spinner=False)
def load_base_data():
    try:
        if not os.path.exists("DailyAIResults.zip"): return None, None, None, None, None, None, None, None, None, None
        with zipfile.ZipFile("DailyAIResults.zip", 'r') as z:
            csv_name = [f for f in z.namelist() if f.endswith('.csv')][0]
            with z.open(csv_name) as f:
                df_all = pd.read_csv(f)
        df_all.columns = df_all.columns.str.strip()
        def clean_date(x):
            s = str(x).split('.')[0].strip()
            if len(s) > 6: s = s[-6:]
            return s
        df_all['Date_Key'] = df_all['Date'].apply(clean_date)
        df_all['Date_DT'] = pd.to_datetime(df_all['Date_Key'], format='%y%m%d', errors='coerce')
        
        feats = ['Comb. Rank', 'Comp. Rank', 'Speed Rank', 'Race Rank', '7:30AM Price', 'No. of Rnrs', 'Trainer PRB', 'Jockey PRB', 'Primary Rank', 'Form Rank', 'MSAI Rank']
        shadow_feats = ['Comb. Rank', 'Comp. Rank', 'Speed Rank', 'Race Rank', 'No. of Rnrs', 'Trainer PRB', 'Jockey PRB', 'Primary Rank', 'Form Rank', 'MSAI Rank']
        
        for col in feats + ['Win P/L <2%', 'Place P/L <2%', 'Fin Pos']:
            if col in df_all.columns:
                df_all[col] = pd.to_numeric(df_all[col], errors='coerce').fillna(0).astype(np.float64)

        model_file = "botman_models.pkl"
        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                saved_models = pickle.load(f)
                clf, shadow_clf = saved_models['clf'], saved_models['shadow_clf']
        else:
            clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, max_depth=5, l2_regularization=2.0, random_state=42)
            shadow_clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, max_depth=5, l2_regularization=2.0, random_state=42)
            train_df = df_all[df_all['Fin Pos'] > 0].tail(230000)
            clf.fit(train_df[feats], (train_df['Fin Pos'] == 1).astype(int))
            shadow_clf.fit(train_df[shadow_feats], (train_df['Fin Pos'] == 1).astype(int))
            with open(model_file, 'wb') as f:
                pickle.dump({'clf': clf, 'shadow_clf': shadow_clf}, f)
        
        df_today = pd.read_csv("DailyAIPredictionsData.csv") if os.path.exists("DailyAIPredictionsData.csv") else None
        return clf, feats, shadow_clf, shadow_feats, df_all, df_today
    except Exception as e: return None, str(e), None, None, None, None

# --- NEW: MORNING REFRESH PROCESSING ---
@st.cache_data(show_spinner="Performing Morning Calculations...")
def get_processed_data(_df_all, _df_today, _model, _shadow_model, feats, shadow_feats):
    def process_df(df, is_today=False):
        d = df.copy()
        d.columns = d.columns.str.strip()
        # Essential Probabilities
        d['ML_Prob'] = _model.predict_proba(d[feats].fillna(0))[:, 1]
        d['Shadow_Prob'] = _shadow_model.predict_proba(d[shadow_feats].fillna(0))[:, 1]
        
        # Determine Grouping Keys
        g_keys = ['Time', 'Course'] if is_today else ['Date_Key', 'Time', 'Course']
        
        # Calculation Ranks
        d['Rank'] = d.groupby(g_keys)['ML_Prob'].rank(ascending=False, method='min')
        d['Pure Rank'] = d.groupby(g_keys)['Shadow_Prob'].rank(ascending=False, method='min')
        d['Value Price'] = 1 / d['ML_Prob']
        
        # Mauve Highlight Logic
        if 'No. of Top' in d.columns:
            d['No. of Top'] = pd.to_numeric(d['No. of Top'], errors='coerce').fillna(0)
            d['Max_Top'] = d.groupby(g_keys)['No. of Top'].transform('max')
            d['isM'] = (d['No. of Top'] == d['Max_Top']) & (d['No. of Top'] > 0)
            
        # Standard Ranks
        d['Primary Rank'] = d.groupby(g_keys)['No. of Top'].transform(lambda x: x.rank(ascending=False, method='min'))
        d['Form Rank'] = d.groupby(g_keys)['Total'].transform(lambda x: x.rank(ascending=False, method='min'))
        return d

    p_all = process_df(_df_all)
    p_today = process_df(_df_today, is_today=True) if _df_today is not None else None
    
    # Split historic/live
    split_date = pd.Timestamp(2026, 3, 8)
    p_all['Date_DT'] = pd.to_datetime(p_all['Date_Key'], format='%y%m%d', errors='coerce')
    df_h = p_all[p_all['Date_DT'] <= split_date].copy()
    
    df_live = None
    if os.path.exists("BOTManAIPredictionsMaster.ods"):
        df_ods = pd.read_excel("BOTManAIPredictionsMaster.ods", engine="odf")
        df_ods.columns = df_ods.columns.str.strip()
        df_ods['Date_Key'] = df_ods['Date'].astype(str).str.split('.').str[0].str.strip().str[-6:]
        ods_keys = df_ods[['Date_Key', 'Time', 'Course', 'Horse', 'Rank']].copy()
        live_pool = p_all[p_all['Date_DT'] > split_date]
        df_live = pd.merge(ods_keys, live_pool, on=['Date_Key', 'Time', 'Course', 'Horse'], how='inner')

    return df_h, df_live, p_today, p_all

# Execute Data Engine
model, feats, shadow_model, shadow_feats, raw_all, raw_today = load_base_data()
df_hist, df_live, df_today, df_all = get_processed_data(raw_all, raw_today, model, shadow_model, feats, shadow_feats)

last_live_date = df_live['Date_DT'].max() if (df_live is not None and not df_live.empty) else datetime.now()
first_res_date = df_hist['Date_DT'].min() if not df_hist.empty else datetime(2024,1,1)

# --- 4. CSS (UNCHANGED) ---
st.markdown('<style>.block-container { padding-top: 1.5rem !important; } header { visibility: hidden; } .scrollable-table { width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch; margin-bottom: 10px; border-radius: 4px; } .k2-table { border-collapse: collapse !important; width: 100% !important; min-width: 800px; table-layout: fixed !important; margin-bottom: 0px !important; } .k2-table th, .k2-table td { border: 1px solid #444 !important; padding: 3px 4px !important; font-size: 12.5px !important; white-space: nowrap !important; overflow: hidden !important; text-overflow: ellipsis !important; } .k2-table td.r1 { background-color: #2e7d32 !important; color: white !important; font-weight: bold !important; } .k2-table td.r2 { background-color: #fbc02d !important; color: black !important; font-weight: bold !important; } .k2-table td.r3 { background-color: #1976d2 !important; color: white !important; font-weight: bold !important; } .mauve-row td { background-color: #f3e5f5 !important; color: black !important; } .k2-table tr:hover td { background-color: #aec6cf !important; color: black !important; } .k2-table thead th { background-color: #000 !important; color: white !important; text-transform: uppercase; letter-spacing: 0.5px; } .left-head { text-align: left !important; padding-left: 10px !important; } .left-text { text-align: left !important; padding-left: 10px !important; } .center-text { text-align: center !important; } .pos-val { color: #2e7d32 !important; font-weight: bold !important; } .neg-val { color: #d32f2f !important; font-weight: bold !important; }</style>', unsafe_allow_html=True)

# --- 5. SIDEBAR NAVIGATION ---
with st.sidebar:
    if os.path.exists("BOTManLogo.png"):
        st.image("BOTManLogo.png")
    st.title("BOTMan Menu")
    app_mode = st.radio("Navigate To:", ["📅 Daily Predictions", "📊 AI Top 2 Results", "🧠 General Systems", "🛠️ System Builder", "🏇 Race Analysis"])
    
    st.markdown("---")
    if st.session_state.get("is_admin"):
        if st.button("⚡ Daily Refresh", use_container_width=True):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()
        if st.button("🧠 Retrain AI", use_container_width=True):
            if os.path.exists("botman_models.pkl"): os.remove("botman_models.pkl")
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()
        admin_ins = st.checkbox("Show Admin Insights", value=st.session_state.get("show_admin_insights", False))
        st.session_state.show_admin_insights = admin_ins

# --- 6. HEADER ---
res_str = last_live_date.strftime('%d %b %Y').upper() if last_live_date else "08 MAR 2026"
st.markdown(f'<div style="display:flex; align-items:center; gap:20px; background-color:#1a3a5f; padding:15px; border-radius:10px; color:white;"><div><div style="font-size:24px; font-weight:bold;">BOTMan Betting Systems</div><div style="margin-top:5px;"><span style="background:#2e7d32; color:white; padding:2px 8px; border-radius:10px; font-size:12px;">✅ LIVE RESULTS TO {res_str}</span></div></div></div>', unsafe_allow_html=True)

# --- 7. ROUTING LOGIC ---

if st.session_state.get("is_admin") and st.session_state.get("show_admin_insights"):
    # [Insert your existing ADMIN INSIGHTS code block here, utilizing 'df_all']
    st.header("🔍 Admin Data Insights")
    pass

elif app_mode == "📅 Daily Predictions":
    # [Insert your Tab 1 logic here, but use pre-processed 'df_today']
    if df_today is not None:
        csv_data = df_today[df_today['Rank'] <= 2].to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv_data, f"BOTMan_Predictions_{datetime.now().strftime('%H%M%S')}.csv", key="dl_p")
        # ... rest of your table display ...
    pass

elif app_mode == "📊 AI Top 2 Results":
    # [Insert Tab 2 logic here]
    pass

elif app_mode == "🏇 Race Analysis":
    st.header("🏇 Race Analysis")
    if df_today is not None:
        # Check if a race is selected
        if st.session_state.get('analysis_race'):
            sel_c, sel_t = st.session_state.analysis_race['course'], st.session_state.analysis_race['time']
            # NAVIGATION BAR (PREV/NEXT)
            # ... insert your existing nav buttons logic ...
            
            # THE TABLE (LIGHTNING FAST FILTRATION)
            race_info = df_today[(df_today['Course'] == sel_c) & (df_today['Time'] == sel_t)]
            # ... insert your existing HTML table construction logic ...
            
        else:
            # GRID SELECTION
            courses = sorted(df_today['Course'].unique())
            for course in courses:
                st.markdown(f"**{course}**")
                races = df_today[df_today['Course'] == course][['Time', 'Race Type', 'H/Cap']].drop_duplicates()
                cols = st.columns(10)
                for i, (_, r) in enumerate(races.iterrows()):
                    if cols[i%10].button(f"{r['Time']}", key=f"btn_{course}_{r['Time']}"):
                        st.session_state.analysis_race = {'course': course, 'time': r['Time']}
                        st.rerun()

# [Keep your remaining Builder/Systems logic sections here]
