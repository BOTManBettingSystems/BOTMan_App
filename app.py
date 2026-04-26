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
import uuid
from datetime import datetime, timedelta
import extra_streamlit_components as stx

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="BOTMan Betting Systems", page_icon="BOTManLogo.png", layout="wide", initial_sidebar_state="expanded")
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:6] 

# --- 2. ACCESS CONTROL ---
def check_password():
    cookie_manager = stx.CookieManager()
    auth_cookie = cookie_manager.get(cookie="botman_auth")
    
    if auth_cookie in ["Admin", "Guest"]:
        st.session_state["password_correct"] = True
        st.session_state["is_admin"] = (auth_cookie == "Admin")
        return True
        
    def password_entered():
        admin_p = st.secrets["ADMIN_PASSWORD"]
        guest_p = st.secrets["GUEST_PASSWORD"]
        entered = st.session_state.get("password_input", "")
        
        if entered in [admin_p, guest_p]:
            st.session_state["password_correct"] = True
            st.session_state["is_admin"] = (entered == admin_p)
            user_type = "Admin" if st.session_state["is_admin"] else "Guest"
            expire_date = datetime.now() + timedelta(days=30)
            cookie_manager.set("botman_auth", user_type, expires_at=expire_date)
            
            with open("login_history.csv", "a") as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{user_type},Session:{st.session_state.session_id}\n")
                
            if "password_input" in st.session_state:
                del st.session_state["password_input"]
        else:
            st.session_state["password_correct"] = False

    if not st.session_state.get("password_correct", False):
        st.markdown("<h2 style='text-align: center; color: #002147;'>BOTMan Betting Systems</h2>", unsafe_allow_html=True)
        st.text_input("Enter Password", type="password", on_change=password_entered, key="password_input")
        return False
        
    return True

if not check_password(): st.stop()

# --- 3. DATA ENGINE ---
@st.cache_resource(show_spinner=False)
def load_all_data():
    try:
        if not os.path.exists("DailyAIResults.zip"): 
            return [None]*11
        
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
        
        df_all['No. of Top'] = pd.to_numeric(df_all.get('No. of Top', 0), errors='coerce').fillna(0)
        df_all['Total'] = pd.to_numeric(df_all.get('Total', 0), errors='coerce').fillna(0)
        df_all['Primary Rank'] = df_all.groupby(['Date_Key', 'Time', 'Course'])['No. of Top'].transform(lambda x: x.rank(ascending=False, method='min'))
        df_all['Form Rank'] = df_all.groupby(['Date_Key', 'Time', 'Course'])['Total'].transform(lambda x: x.rank(ascending=False, method='min'))
        if 'MSAI Rank' not in df_all.columns: df_all['MSAI Rank'] = 0
        df_all['MSAI Rank'] = pd.to_numeric(df_all['MSAI Rank'], errors='coerce').fillna(0)

        feats = ['Comb. Rank', 'Comp. Rank', 'Speed Rank', 'Race Rank', '7:30AM Price', 'No. of Rnrs', 'Trainer PRB', 'Jockey PRB', 'Primary Rank', 'Form Rank', 'MSAI Rank']
        shadow_feats = ['Comb. Rank', 'Comp. Rank', 'Speed Rank', 'Race Rank', 'No. of Rnrs', 'Trainer PRB', 'Jockey PRB', 'Primary Rank', 'Form Rank', 'MSAI Rank']
        
        for col in feats + ['Win P/L <2%', 'Place P/L <2%', 'Fin Pos']:
            if col in df_all.columns:
                df_all[col] = pd.to_numeric(df_all[col], errors='coerce').fillna(0).astype(np.float64)

        split_date = pd.Timestamp(2026, 3, 8)
        df_historic = df_all[df_all['Date_DT'] <= split_date].copy()
        df_live = None
        
        if os.path.exists("BOTManAIPredictionsMaster.ods"):
            df_ods = pd.read_excel("BOTManAIPredictionsMaster.ods", engine="odf")
            df_ods.columns = df_ods.columns.str.strip()
            df_ods['Date_Key'] = df_ods['Date'].apply(clean_date)
            ods_keys = df_ods[['Date_Key', 'Time', 'Course', 'Horse', 'Rank']].copy()
            live_res_pool = df_all[df_all['Date_DT'] > split_date]
            df_live = pd.merge(ods_keys, live_res_pool, on=['Date_Key', 'Time', 'Course', 'Horse'], how='inner')
                
        # --- MODEL LOADING (VERSION-SAFE FOR HUGGING FACE) ---
        model_file = "botman_models.pkl"
        force_retrain = False
        
        if os.path.exists(model_file):
            try:
                with open(model_file, 'rb') as f:
                    saved_models = pickle.load(f)
                    # CHECK: Does this file have the 3rd brain?
                    if 'cal_clf' in saved_models:
                        clf = saved_models['clf']
                        shadow_clf = saved_models['shadow_clf']
                        cal_clf = saved_models['cal_clf']
                    else:
                        force_retrain = True # Old 2-brain file found, ignore it
            except Exception:
                force_retrain = True
        else:
            force_retrain = True

        if force_retrain:
            with st.spinner("🧠 Birthing the Triple-Brain Engine... Please wait (60s)"):
                clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, max_depth=5, l2_regularization=2.0, random_state=42)
                shadow_clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, max_depth=5, l2_regularization=2.0, random_state=42)
                
                # THE LEASHED PRICER
                cal_clf = HistGradientBoostingClassifier(
                    max_iter=100, 
                    learning_rate=0.05, 
                    max_depth=3, 
                    l2_regularization=15.0, 
                    min_samples_leaf=250, 
                    random_state=42
                )
                
                train_df = df_all[df_all['Fin Pos'] > 0].tail(230000)
                target = (train_df['Fin Pos'] == 1).astype(int)
                
                clf.fit(train_df[feats], target)
                shadow_clf.fit(train_df[shadow_feats], target)
                cal_clf.fit(train_df[feats], target)
                
                # Save the new 3-brain version
                with open(model_file, 'wb') as f:
                    pickle.dump({'clf': clf, 'shadow_clf': shadow_clf, 'cal_clf': cal_clf}, f)
                gc.collect()
        
        df_today = pd.read_csv("DailyAIPredictionsData.csv") if os.path.exists("DailyAIPredictionsData.csv") else None
        if df_today is not None:
            df_today.columns = df_today.columns.str.strip()
            df_today['No. of Top'] = pd.to_numeric(df_today.get('No. of Top', 0), errors='coerce').fillna(0)
            df_today['Total'] = pd.to_numeric(df_today.get('Total', 0), errors='coerce').fillna(0)
            df_today['Primary Rank'] = df_today.groupby(['Time', 'Course'])['No. of Top'].transform(lambda x: x.rank(ascending=False, method='min'))
            df_today['Form Rank'] = df_today.groupby(['Time', 'Course'])['Total'].transform(lambda x: x.rank(ascending=False, method='min'))
            if 'MSAI Rank' not in df_today.columns: df_today['MSAI Rank'] = 0
            df_today['MSAI Rank'] = pd.to_numeric(df_today['MSAI Rank'], errors='coerce').fillna(0)
            
            # Pre-calculate probabilities for today
            df_today['ML_Prob'] = clf.predict_proba(df_today[feats].fillna(0))[:, 1]
            df_today['Rank'] = df_today.groupby(['Time', 'Course'])['ML_Prob'].rank(ascending=False, method='min')
            df_today['Value Price'] = 1 / df_today['ML_Prob']
            
            # Calibrated Value for Today
            df_today['True_AI_Prob'] = cal_clf.predict_proba(df_today[feats].fillna(0))[:, 1]
            df_today['True_Value_Price'] = np.where(df_today['True_AI_Prob'] > 0.001, 1.0 / df_today['True_AI_Prob'], 1000.0)

            # Pure Rank
            df_today['Shadow_Prob'] = shadow_clf.predict_proba(df_today[shadow_feats].fillna(0))[:, 1]
            df_today['Pure Rank'] = df_today.groupby(['Time', 'Course'])['Shadow_Prob'].rank(ascending=False, method='min')
            
            df_today['Rank2_Prob'] = df_today.groupby(['Time', 'Course'])['ML_Prob'].transform(lambda x: x.nlargest(2).iloc[-1] if len(x) > 1 else 0)
            df_today['Prob Gap'] = df_today['ML_Prob'] - df_today['Rank2_Prob']
                
        last_live = df_live['Date_DT'].max() if (df_live is not None and not df_live.empty) else datetime.now()
        first_hist = df_historic['Date_DT'].min() if not df_historic.empty else datetime(2024,1,1)
        
        return clf, feats, shadow_clf, shadow_feats, cal_clf, df_historic, df_live, df_today, last_live, first_hist, df_all
    except Exception as e: 
        print(f"Error loading data: {e}")
        return [None]*11

# --- 6. OPTIMIZATION ENGINE ---
@st.cache_data(show_spinner=False)
def prep_system_builder_data(_df, _model, feats, _shadow_model=None, shadow_feats=None, _cal_model=None, is_live_today=False, use_vault=True):
    b_df = _df.copy()
    b_df.columns = b_df.columns.str.strip()
    
    # Basic date prep
    if 'Date_Key' not in b_df.columns and 'Date' in b_df.columns:
        b_df['Date_Key'] = b_df['Date'].astype(str).str.split('.').str[0].str.strip()
        b_df['Date_Key'] = b_df['Date_Key'].apply(lambda s: s[-6:] if len(s) > 6 else s)
    if 'Date_DT' not in b_df.columns and 'Date_Key' in b_df.columns:
        b_df['Date_DT'] = pd.to_datetime(b_df['Date_Key'], format='%y%m%d', errors='coerce')
    
    if not is_live_today:
        b_df = b_df[b_df.get('Fin Pos', 0) > 0].copy()
    
    # --- VAULT LOGIC ---
    if os.path.exists("BOTMan_Prediction_Vault.csv") and not is_live_today and use_vault:
        vault_df = pd.read_csv("BOTMan_Prediction_Vault.csv")
        for c in ['Date', 'Time', 'Course', 'Horse']:
            if c in vault_df.columns and c in b_df.columns:
                vault_df[c] = vault_df[c].astype(str).str.strip()
                b_df[c] = b_df[c].astype(str).str.strip()
        v_sub = vault_df[['Date', 'Time', 'Course', 'Horse', 'ML_Prob', 'Rank', 'Value Price']].rename(
            columns={'ML_Prob': 'ML_Prob_vault', 'Rank': 'Rank_vault', 'Value Price': 'Value Price_vault'}
        )
        b_df = pd.merge(b_df, v_sub, on=['Date', 'Time', 'Course', 'Horse'], how='left')
        missing_mask = b_df['ML_Prob_vault'].isna()
        if missing_mask.any():
            new_probs = _model.predict_proba(b_df[missing_mask][feats].fillna(0))[:, 1]
            b_df.loc[missing_mask, 'ML_Prob'] = new_probs
            b_df['ML_Prob'] = b_df['ML_Prob_vault'].fillna(b_df.get('ML_Prob'))
            b_df['Rank'] = b_df['Rank_vault'].fillna(b_df.groupby(['Date_Key', 'Time', 'Course'])['ML_Prob'].rank(ascending=False, method='min'))
            b_df['Value Price'] = b_df['Value Price_vault'].fillna(1 / b_df['ML_Prob'])
        else:
            b_df['ML_Prob'], b_df['Rank'], b_df['Value Price'] = b_df['ML_Prob_vault'], b_df['Rank_vault'], b_df['Value Price_vault']
        b_df = b_df.drop(columns=['ML_Prob_vault', 'Rank_vault', 'Value Price_vault'])
    else:
        b_df['ML_Prob'] = _model.predict_proba(b_df[feats].fillna(0))[:, 1]
        b_df['Rank'] = b_df.groupby(['Date_Key', 'Time', 'Course'])['ML_Prob'].rank(ascending=False, method='min')
        b_df['Value Price'] = 1 / b_df['ML_Prob']

    # --- CALIBRATED BRAIN INTEGRATION (TRUE VALUE & EDGE) ---
    if _cal_model is not None:
        b_df['True_AI_Prob'] = _cal_model.predict_proba(b_df[feats].fillna(0))[:, 1]
        b_df['Cal_Value_Price'] = np.where(b_df['True_AI_Prob'] > 0.001, 1.0 / b_df['True_AI_Prob'], 1000.0)
        
        # Calculate Edge against the Leashed model
        market_p = np.where(b_df['BSP'] > 0, b_df['BSP'], b_df['7:30AM Price'])
        b_df['Value_Edge_Perc'] = ((market_p / b_df['Cal_Value_Price']) - 1) * 100
        
        # Edge Brackets for X-Ray
        v_bins = [-np.inf, 0.0, 10.0, 20.0, np.inf]
        v_labels = ['1. Negative Edge (<0%)', '2. Fair Value (0-10%)', '3. Value (10-20%)', '4. Deep Value (>20%)']
        b_df['Edge Bracket'] = pd.cut(b_df['Value_Edge_Perc'], bins=v_bins, labels=v_labels)
        b_df['Edge Bracket'] = b_df['Edge Bracket'].cat.add_categories('Unknown').fillna('Unknown')
    else:
        b_df['Value_Edge_Perc'] = 0.0
        b_df['Edge Bracket'] = 'Unknown'
        b_df['Cal_Value_Price'] = b_df['Value Price']

    # Shadow Model / Pure Rank
    if _shadow_model is not None:
        b_df['Shadow_Prob'] = _shadow_model.predict_proba(b_df[shadow_feats].fillna(0))[:, 1]
        b_df['Pure Rank'] = b_df.groupby(['Date_Key', 'Time', 'Course'])['Shadow_Prob'].rank(ascending=False, method='min')
    
    # Secondary cleanup
    b_df['7:30AM Price'] = pd.to_numeric(b_df.get('7:30AM Price', 0), errors='coerce')
    b_df['BSP'] = pd.to_numeric(b_df.get('BSP', 0), errors='coerce')
    if not is_live_today:
        b_df['Win P/L <2%'] = pd.to_numeric(b_df.get('Win P/L <2%', 0), errors='coerce')
        b_df['Is_Win'] = np.where(b_df['Win P/L <2%'] > 0, 1, 0)
        b_df['Is_Place'] = np.where((b_df['Fin Pos'] >= 1) & (b_df['Fin Pos'] <= 3), 1, 0)
        
    b_df['Rank2_Prob'] = b_df.groupby(['Date_Key', 'Time', 'Course'])['ML_Prob'].transform(lambda x: x.nlargest(2).iloc[-1] if len(x) > 1 else 0)
    b_df['Prob Gap'] = b_df['ML_Prob'] - b_df['Rank2_Prob']
    
    bins = [-1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0, 50.0, 1000.0]
    labels = ["<2.0", "2.01-3.0", "3.01-4.0", "4.01-5.0", "5.01-6.0", "6.01-7.0", "7.01-8.0", "8.01-9.0", "9.01-10.0", "10.01-15.0", "15.01-20.0", "20.01-50.0", "50.01+"]
    b_df['Price Bracket'] = pd.cut(b_df['7:30AM Price'], bins=bins, labels=labels, right=True)
    
    return b_df

# --- 5. EXECUTION ---
model, feats, shadow_model, shadow_feats, cal_model, df_hist, df_live, df_today, last_live_date, first_res_date, df_all = load_all_data()

if model is None:
    st.error("🚨 Critical Error: Could not load data.")
    st.stop()

# --- Page 4: Mini SYSTEM BUILDER ---
if "app_mode" not in st.session_state: st.session_state.app_mode = "🛠️ System Builder" # Default for code display
app_mode = st.session_state.app_mode

if app_mode == "🛠️ System Builder":
    if "form_reset_counter" not in st.session_state: st.session_state.form_reset_counter = 0
    if "sys_defaults" not in st.session_state: st.session_state.sys_defaults = {}

    st.header("🛠️ Mini System Builder")
    
    if st.session_state.get("is_admin"):
        ai_mode = st.radio("🧠 **AI Backtest Engine:**", ["💾 Use Prediction Vault", "⚡ Use Today's Live Brain"], horizontal=True)
        use_vault_bool = "Vault" in ai_mode
    else:
        use_vault_bool = True

    if df_all is not None:
        # Pass the 3rd model into the prep engine
        b_df = prep_system_builder_data(df_all, model, feats, shadow_model, shadow_feats, cal_model, use_vault=use_vault_bool)

        with st.form(f"builder_form_{st.session_state.form_reset_counter}"):
            defs = st.session_state.sys_defaults
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                race_types = b_df['Race Type'].dropna().unique().tolist()
                selected_race_types = st.multiselect("Race Type", race_types, default=defs.get('race_types', race_types))
                price_min = st.number_input("Min Price", 0.0, 1000.0, float(defs.get('price_min', 0.0)))
            with c2:
                price_max = st.number_input("Max Price", 0.0, 1000.0, float(defs.get('price_max', 1000.0)))
                # THE NEW CALIBRATED EDGE SLIDER
                min_edge_perc = st.number_input("Min Value Edge % (Leashed)", -100.0, 500.0, float(defs.get('min_edge_perc', -100.0)), step=5.0, help="Filters bets based on the Leashed Brain's margin over the market price.")
            with c3:
                vf_opts = ["Off", "Original AI vs 7:30AM", "Original AI vs BSP", "Calibrated AI (Leashed) vs 7:30AM", "Calibrated AI (Leashed) vs BSP"]
                value_filter = st.selectbox("Value Strategy", vf_opts, index=0)
            with c4:
                selected_groupby = st.multiselect("Group By", ['Race Type', 'H/Cap', 'Price Bracket', 'Edge Bracket', 'Course'], default=defs.get('groupby', ['Race Type', 'H/Cap']))

            submit_button = st.form_submit_button(label="🚀 Process Data")

        if submit_button:
            # --- APPLY FILTERS ---
            mask = (b_df['Race Type'].isin(selected_race_types) & 
                    (b_df['7:30AM Price'] >= price_min) & 
                    (b_df['7:30AM Price'] <= price_max) &
                    (b_df['Value_Edge_Perc'] >= min_edge_perc))
            
            # Dynamic Value Filter Logic
            if "Original AI" in value_filter:
                v_col = 'Value Price'
                m_col = 'BSP' if "BSP" in value_filter else '7:30AM Price'
                mask &= (b_df[m_col] > b_df[v_col])
            elif "Calibrated AI" in value_filter:
                v_col = 'Cal_Value_Price'
                m_col = 'BSP' if "BSP" in value_filter else '7:30AM Price'
                mask &= (b_df[m_col] > b_df[v_col])

            df_filtered = b_df[mask].copy()

            if not df_filtered.empty:
                # Grouping and KPI calculations
                breakdown = df_filtered.groupby(selected_groupby, observed=False).agg(
                    Bets=('Horse', 'count'), Wins=('Is_Win', 'sum'), Profit=('Win P/L <2%', 'sum')
                ).reset_index()
                
                total_bets = breakdown['Bets'].sum()
                total_wins = breakdown['Wins'].sum()
                total_profit = breakdown['Profit'].sum()
                roi = (total_profit / total_bets * 100) if total_bets > 0 else 0
                
                # Metrics Row
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total Bets", total_bets)
                k2.metric("Wins", total_wins)
                k3.metric("Win P/L", f"£{total_profit:.2f}")
                k4.metric("ROI", f"{roi:.1f}%")

                # --- NEW: ELITE QUANT METRICS BLOCK ---
                chron_df = df_filtered.sort_values(by=['Date_DT', 'Time'])
                
                # A/E Ratio
                exp_wins = (1 / chron_df['7:30AM Price']).sum()
                ae_ratio = total_wins / exp_wins if exp_wins > 0 else 0
                
                # Full Chi-Square (Wins + Losses)
                if exp_wins > 0 and total_bets > exp_wins:
                    exp_losses = total_bets - exp_wins
                    act_losses = total_bets - total_wins
                    chi = ((total_wins - exp_wins)**2 / exp_wins) + ((act_losses - exp_losses)**2 / exp_losses)
                else: chi = 0
                
                # Daily Sortino
                daily = chron_df.groupby(chron_df['Date_DT'].dt.date)['Win P/L <2%'].sum()
                d_std = daily[daily < 0].std() if not daily[daily < 0].empty else 0.001
                sortino = (daily.mean() / d_std) * (365**0.5) if d_std > 0 else 0
                
                # Ulcer Index
                cum_p = chron_df['Win P/L <2%'].cumsum()
                dd = cum_p.cummax() - cum_p
                ulcer = (dd**2).mean()**0.5

                q1, q2, q3, q4 = st.columns(4)
                q1.metric("A/E Ratio", f"{ae_ratio:.2f}", help="Actual vs Expected wins. >1.00 is beating the market.")
                q2.metric("Chi Score", f"{chi:.1f}", help="Statistical significance. >6.6 is 99% certain it's not luck.")
                q3.metric("Sortino", f"{sortino:.2f}", help="Annualized risk-adjusted return (Daily). >2.0 is elite.")
                q4.metric("Ulcer Index", f"{ulcer:.1f}", help="Average drawdown depth in pts. <10 is low stress.")

                st.dataframe(breakdown, use_container_width=True)
            else:
                st.warning("No matches found.")
