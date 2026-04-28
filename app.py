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
    st.session_state.session_id = str(uuid.uuid4())[:6] # Generates a random 6-letter code

# --- 2. ACCESS CONTROL ---
def check_password():
    # Initialize the Cookie Manager
    cookie_manager = stx.CookieManager()
    
    # 1. Check if the user already has a valid 30-day cookie
    auth_cookie = cookie_manager.get(cookie="botman_auth")
    
    if auth_cookie in ["Admin", "Guest"]:
        st.session_state["password_correct"] = True
        st.session_state["is_admin"] = (auth_cookie == "Admin")
        return True
        
    # 2. If no cookie, show the standard login form
    def password_entered():
        admin_p = st.secrets["ADMIN_PASSWORD"]
        guest_p = st.secrets["GUEST_PASSWORD"]
        entered = st.session_state.get("password_input", "")
        
        if entered in [admin_p, guest_p]:
            st.session_state["password_correct"] = True
            st.session_state["is_admin"] = (entered == admin_p)
            user_type = "Admin" if st.session_state["is_admin"] else "Guest"
            
            # --- THE MAGIC: Set a persistent cookie for 30 days ---
            expire_date = datetime.now() + timedelta(days=30)
            cookie_manager.set("botman_auth", user_type, expires_at=expire_date)
            
            # Log the new fresh session
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
                
        # --- MODEL LOADING (HUGGING FACE VERSION-SAFE DOUBLE BRAIN) ---
        model_file = "botman_models.pkl"
        force_retrain = False
        cal_clf = None
        
        if os.path.exists(model_file):
            try:
                with open(model_file, 'rb') as f:
                    saved_models = pickle.load(f)
                    if 'cal_clf' in saved_models:
                        clf = saved_models['clf']
                        shadow_clf = saved_models['shadow_clf']
                        cal_clf = saved_models['cal_clf']
                    else:
                        force_retrain = True # Old file found, missing 3rd brain
            except Exception:
                force_retrain = True
        else:
            force_retrain = True

        if force_retrain:
            clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, max_depth=5, l2_regularization=2.0, random_state=42)
            shadow_clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, max_depth=5, l2_regularization=2.0, random_state=42)
            
            # THE LEASHED PRICER: High L2, shallow depth, large leaf requirements
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
            gc.collect()
            
            with open(model_file, 'wb') as f:
                pickle.dump({'clf': clf, 'shadow_clf': shadow_clf, 'cal_clf': cal_clf}, f)
        
        df_today = pd.read_csv("DailyAIPredictionsData.csv") if os.path.exists("DailyAIPredictionsData.csv") else None
        if df_today is not None:
            df_today.columns = df_today.columns.str.strip()
            df_today['No. of Top'] = pd.to_numeric(df_today.get('No. of Top', 0), errors='coerce').fillna(0)
            df_today['Total'] = pd.to_numeric(df_today.get('Total', 0), errors='coerce').fillna(0)
            df_today['Primary Rank'] = df_today.groupby(['Time', 'Course'])['No. of Top'].transform(lambda x: x.rank(ascending=False, method='min'))
            df_today['Form Rank'] = df_today.groupby(['Time', 'Course'])['Total'].transform(lambda x: x.rank(ascending=False, method='min'))
            if 'MSAI Rank' not in df_today.columns: df_today['MSAI Rank'] = 0
            df_today['MSAI Rank'] = pd.to_numeric(df_today['MSAI Rank'], errors='coerce').fillna(0)
            
            # THE SPEED FIX: Pre-calculate everything here so tabs load instantly
            missing_feats = [f for f in feats if f not in df_today.columns]
            if not missing_feats:
                df_today['ML_Prob'] = clf.predict_proba(df_today[feats].fillna(0))[:, 1]
                df_today['Rank'] = df_today.groupby(['Time', 'Course'])['ML_Prob'].rank(ascending=False, method='min')
                df_today['Value Price'] = 1 / df_today['ML_Prob']
                
                # Calibrated Value for Today
                if cal_clf is not None:
                    df_today['True_AI_Prob'] = cal_clf.predict_proba(df_today[feats].fillna(0))[:, 1]
                    df_today['Cal_Value_Price'] = np.where(df_today['True_AI_Prob'] > 0.001, 1.0 / df_today['True_AI_Prob'], 1000.0)

                # Pre-calculate pure ranks using the shadow model
                missing_shadow = [f for f in shadow_feats if f not in df_today.columns]
                if not missing_shadow:
                    df_today['Shadow_Prob'] = shadow_clf.predict_proba(df_today[shadow_feats].fillna(0))[:, 1]
                    df_today['Pure Rank'] = df_today.groupby(['Time', 'Course'])['Shadow_Prob'].rank(ascending=False, method='min')
                
                # Pre-calculate the gap for value finding
                df_today['Rank2_Prob'] = df_today.groupby(['Time', 'Course'])['ML_Prob'].transform(lambda x: x.nlargest(2).iloc[-1] if len(x) > 1 else 0)
                df_today['Prob Gap'] = df_today['ML_Prob'] - df_today['Rank2_Prob']
                
        last_live = df_live['Date_DT'].max() if (df_live is not None and not df_live.empty) else datetime.now()
        first_hist = df_historic['Date_DT'].min() if not df_historic.empty else datetime(2024,1,1)
        
        return clf, feats, shadow_clf, shadow_feats, cal_clf, df_historic, df_live, df_today, last_live, first_hist, df_all
    except Exception as e: 
        print(f"Error loading data: {e}")
        return [None]*11

@st.cache_data(show_spinner=False)
def load_ods_master():
    if os.path.exists("BOTManSystemsMaster.ods"):
        return pd.read_excel("BOTManSystemsMaster.ods", engine="odf")
    return None
    
# --- 6. OPTIMIZATION ENGINE FOR TAB 4 & ADMIN ---
@st.cache_data(show_spinner=False)
def prep_system_builder_data(_df, _model, feats, _shadow_model=None, shadow_feats=None, _cal_model=None, is_live_today=False, use_vault=True):
    b_df = _df.copy()
    b_df.columns = b_df.columns.str.strip()
    
    if 'Date_Key' not in b_df.columns and 'Date' in b_df.columns:
        b_df['Date_Key'] = b_df['Date'].astype(str).str.split('.').str[0].str.strip()
        b_df['Date_Key'] = b_df['Date_Key'].apply(lambda s: s[-6:] if len(s) > 6 else s)
    
    if 'Date_DT' not in b_df.columns and 'Date_Key' in b_df.columns:
        b_df['Date_DT'] = pd.to_datetime(b_df['Date_Key'], format='%y%m%d', errors='coerce')
        
    if not is_live_today:
        b_df = b_df[b_df.get('Fin Pos', 0) > 0].copy()

    # --- 🛡️ CRITICAL FIX: THE EMPTY DATAFRAME SHIELD ---
    if b_df.empty:
        for col in ['ML_Prob', 'Rank', 'Value Price', 'True_AI_Prob', 'Cal_Value_Price', 'Value_Edge_Perc', 'Shadow_Prob', 'Pure Rank', 'Rank2_Prob', 'Prob Gap', 'User Value']:
            if col not in b_df.columns:
                b_df[col] = 0.0
        b_df['Edge Bracket'] = 'Unknown'
        return b_df
    # ---------------------------------------------------

    # --- THE PREDICTION VAULT BRIDGE ---
    # 🛡️ NEW: Auto-Unzip the Vault if the compressed version is uploaded
    if os.path.exists("BOTMan_Prediction_Vault.zip") and not os.path.exists("BOTMan_Prediction_Vault.csv"):
        try:
            with zipfile.ZipFile("BOTMan_Prediction_Vault.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
        except:
            pass

    if os.path.exists("BOTMan_Prediction_Vault.csv") and not is_live_today and use_vault:
        try:
            vault_df = pd.read_csv("BOTMan_Prediction_Vault.csv")
            if not all(c in vault_df.columns for c in ['Date', 'Time', 'Course', 'Horse']):
                vault_df = pd.DataFrame()
        except:
            vault_df = pd.DataFrame()

        if not vault_df.empty:
            for c in ['Date', 'Time', 'Course', 'Horse']:
                if c in vault_df.columns and c in b_df.columns:
                    vault_df[c] = vault_df[c].astype(str).str.strip()
                    b_df[c] = b_df[c].astype(str).str.strip()
                    
            has_leashed_vault = 'True_AI_Prob' in vault_df.columns
            
            rename_dict = {'ML_Prob': 'ML_Prob_vault', 'Rank': 'Rank_vault', 'Value Price': 'Value Price_vault'}
            if has_leashed_vault:
                rename_dict.update({'True_AI_Prob': 'True_AI_Prob_vault', 'Cal_Value_Price': 'Cal_Value_Price_vault'})
                
            v_cols = ['Date', 'Time', 'Course', 'Horse'] + list(rename_dict.keys())
            
            available_v_cols = [c for c in v_cols if c in vault_df.columns]
            if all(k in available_v_cols for k in ['Date', 'Time', 'Course', 'Horse']):
                v_sub = vault_df[available_v_cols].rename(columns=rename_dict)
                b_df = pd.merge(b_df, v_sub, on=['Date', 'Time', 'Course', 'Horse'], how='left')
            else:
                vault_df = pd.DataFrame() 
        
        if vault_df.empty or 'ML_Prob_vault' not in b_df.columns:
            missing_mask = pd.Series(True, index=b_df.index)
        else:
            missing_mask = b_df['ML_Prob_vault'].isna()
            
        if missing_mask.any():
            new_horses = b_df[missing_mask].copy()
            new_probs = _model.predict_proba(new_horses[feats].fillna(0))[:, 1]
            
            if 'ML_Prob' not in b_df.columns: b_df['ML_Prob'] = np.nan
            b_df.loc[missing_mask, 'ML_Prob'] = new_probs
            
            if 'ML_Prob_vault' in b_df.columns:
                b_df['ML_Prob'] = b_df['ML_Prob_vault'].fillna(b_df['ML_Prob'])
                
            if 'Rank_vault' in b_df.columns:
                b_df['Rank'] = b_df['Rank_vault'].fillna(b_df.groupby(['Date_Key', 'Time', 'Course'])['ML_Prob'].rank(ascending=False, method='min'))
            else:
                b_df['Rank'] = b_df.groupby(['Date_Key', 'Time', 'Course'])['ML_Prob'].rank(ascending=False, method='min')
                
            if 'Value Price_vault' in b_df.columns:
                b_df['Value Price'] = b_df['Value Price_vault'].fillna(1 / b_df['ML_Prob'])
            else:
                b_df['Value Price'] = 1 / b_df['ML_Prob']
            
            if _cal_model is not None:
                new_cal_probs = _cal_model.predict_proba(new_horses[feats].fillna(0))[:, 1]
                
                if 'True_AI_Prob' not in b_df.columns: b_df['True_AI_Prob'] = np.nan
                b_df.loc[missing_mask, 'True_AI_Prob'] = new_cal_probs
                
                if 'True_AI_Prob_vault' in b_df.columns:
                    b_df['True_AI_Prob'] = b_df['True_AI_Prob_vault'].fillna(b_df['True_AI_Prob'])
                    
                if 'Cal_Value_Price_vault' in b_df.columns:
                    b_df['Cal_Value_Price'] = b_df['Cal_Value_Price_vault'].fillna(np.where(b_df['True_AI_Prob'] > 0.001, 1.0 / b_df['True_AI_Prob'], 1000.0))
                else:
                    b_df['Cal_Value_Price'] = np.where(b_df['True_AI_Prob'] > 0.001, 1.0 / b_df['True_AI_Prob'], 1000.0)
            
            append_cols = ['Date', 'Time', 'Course', 'Horse', 'ML_Prob', 'Rank', 'Value Price']
            if _cal_model is not None: append_cols += ['True_AI_Prob', 'Cal_Value_Price']
            
            append_df = b_df[missing_mask][[c for c in append_cols if c in b_df.columns]]
            
            write_header = not os.path.exists("BOTMan_Prediction_Vault.csv") or os.path.getsize("BOTMan_Prediction_Vault.csv") == 0
            append_df.to_csv("BOTMan_Prediction_Vault.csv", mode='a', header=write_header, index=False)
        else:
            b_df['ML_Prob'] = b_df['ML_Prob_vault']
            b_df['Rank'] = b_df['Rank_vault']
            b_df['Value Price'] = b_df['Value Price_vault']
            if 'True_AI_Prob_vault' in b_df.columns:
                b_df['True_AI_Prob'] = b_df['True_AI_Prob_vault']
                b_df['Cal_Value_Price'] = b_df['Cal_Value_Price_vault']
            
        drop_cols = [c for c in ['ML_Prob_vault', 'Rank_vault', 'Value Price_vault', 'True_AI_Prob_vault', 'Cal_Value_Price_vault'] if c in b_df.columns]
        b_df = b_df.drop(columns=drop_cols)
        
    else:
        b_df['ML_Prob'] = _model.predict_proba(b_df[feats].fillna(0))[:, 1]
        b_df['Rank'] = b_df.groupby(['Date_Key', 'Time', 'Course'])['ML_Prob'].rank(ascending=False, method='min')
        b_df['Value Price'] = 1 / b_df['ML_Prob']
    # --- END OF VAULT BRIDGE ---

    # --- CALIBRATED BRAIN INTEGRATION (TRUE VALUE & EDGE) ---
    if _cal_model is not None:
        if 'True_AI_Prob' not in b_df.columns:
            b_df['True_AI_Prob'] = _cal_model.predict_proba(b_df[feats].fillna(0))[:, 1]
            b_df['Cal_Value_Price'] = np.where(b_df['True_AI_Prob'] > 0.001, 1.0 / b_df['True_AI_Prob'], 1000.0)
        
        # 🛡️ NEW: Edge is strictly locked to the 7:30AM Morning Price (ignores BSP entirely)
        safe_morning_price = pd.to_numeric(b_df.get('7:30AM Price', 0), errors='coerce').fillna(0)
        b_df['Value_Edge_Perc'] = np.where(b_df['Cal_Value_Price'] > 0, ((safe_morning_price / b_df['Cal_Value_Price']) - 1) * 100, 0.0)
        
        # Edge Brackets for X-Ray
        v_bins = [-np.inf, 0.0, 10.0, 20.0, np.inf]
        v_labels = ['1. Negative Edge (<0%)', '2. Fair Value (0-10%)', '3. Value (10-20%)', '4. Deep Value (>20%)']
        b_df['Edge Bracket'] = pd.cut(b_df['Value_Edge_Perc'], bins=v_bins, labels=v_labels)
        b_df['Edge Bracket'] = b_df['Edge Bracket'].cat.add_categories('Unknown').fillna('Unknown')
    else:
        b_df['Value_Edge_Perc'] = 0.0
        b_df['Edge Bracket'] = 'Unknown'
        b_df['Cal_Value_Price'] = b_df.get('Value Price', 0.0)
    
    if _shadow_model is not None and shadow_feats is not None:
        missing_shadow = [f for f in shadow_feats if f not in b_df.columns]
        if not missing_shadow:
            b_df['Shadow_Prob'] = _shadow_model.predict_proba(b_df[shadow_feats].fillna(0))[:, 1]
            if is_live_today:
                b_df['Pure Rank'] = b_df.groupby(['Time', 'Course'])['Shadow_Prob'].rank(ascending=False, method='min')
            else:
                b_df['Pure Rank'] = b_df.groupby(['Date_Key', 'Time', 'Course'])['Shadow_Prob'].rank(ascending=False, method='min')
        else:
            b_df['Pure Rank'] = 0
            
    b_df['7:30AM Price'] = pd.to_numeric(b_df.get('7:30AM Price', 0), errors='coerce')
    b_df['BSP'] = pd.to_numeric(b_df.get('BSP', 0), errors='coerce')
    b_df['Age'] = pd.to_numeric(b_df.get('Age', 0), errors='coerce').fillna(0)
    
    b_df['No. of Top'] = pd.to_numeric(b_df.get('No. of Top', 0), errors='coerce').fillna(0)
    b_df['Total'] = pd.to_numeric(b_df.get('Total', 0), errors='coerce').fillna(0)
    
    if not is_live_today:
        b_df['Win P/L <2%'] = pd.to_numeric(b_df.get('Win P/L <2%', 0), errors='coerce')
        b_df['Place P/L <2%'] = pd.to_numeric(b_df.get('Place P/L <2%', 0), errors='coerce')
        b_df['Fin Pos'] = pd.to_numeric(b_df.get('Fin Pos', 0), errors='coerce')
        b_df['Is_Win'] = np.where(b_df['Win P/L <2%'] > 0, 1, 0)
        b_df['Is_Place'] = np.where((b_df['Fin Pos'] >= 1) & (b_df['Fin Pos'] <= 3), 1, 0)
    
    b_df['No. of Rnrs'] = pd.to_numeric(b_df.get('No. of Rnrs', 0), errors='coerce')
    
    b_df['Trainer PRB Rank'] = 0
    b_df['Jockey PRB Rank'] = 0
    b_df['Primary Rank'] = 0
    b_df['Form Rank'] = 0
    
    if 'Trainer PRB' in b_df.columns:
        b_df['Trainer PRB Rank'] = b_df.groupby(['Date_Key', 'Time', 'Course'])['Trainer PRB'].transform(lambda x: x.rank(ascending=False, method='min'))
    if 'Jockey PRB' in b_df.columns:
        b_df['Jockey PRB Rank'] = b_df.groupby(['Date_Key', 'Time', 'Course'])['Jockey PRB'].transform(lambda x: x.rank(ascending=False, method='min'))
        
    b_df['Primary Rank'] = b_df.groupby(['Date_Key', 'Time', 'Course'])['No. of Top'].transform(lambda x: x.rank(ascending=False, method='min'))
    b_df['Form Rank'] = b_df.groupby(['Date_Key', 'Time', 'Course'])['Total'].transform(lambda x: x.rank(ascending=False, method='min'))
    
    b_df['Rank'] = b_df.groupby(['Date_Key', 'Time', 'Course'])['ML_Prob'].rank(ascending=False, method='min')
    b_df['Rank2_Prob'] = b_df.groupby(['Date_Key', 'Time', 'Course'])['ML_Prob'].transform(lambda x: x.nlargest(2).iloc[-1] if len(x) > 1 else 0)
    b_df['Prob Gap'] = b_df['ML_Prob'] - b_df['Rank2_Prob']
    b_df['User Value'] = pd.to_numeric(b_df.get('Value', 0), errors='coerce')
        
    bins = [-1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 15.0, 20.0, 50.0, 100.0, 1000.0]
    labels = ["<1.0", "1.0-2.0", "2.01-3.0", "3.01-4.0", "4.01-5.0", "5.01-6.0", "6.01-7.0", "7.01-8.0", "8.01-9.0", "9.01-10.0", "10.01-11.0", "11.01-15.0", "15.01-20.0", "20.01-50.0", "50.01-100.0", "100.01+"]
    b_df['Price Bracket'] = pd.cut(b_df['7:30AM Price'], bins=bins, labels=labels, right=True)
    b_df['Price Bracket'] = b_df['Price Bracket'].cat.add_categories('Unknown').fillna('Unknown')
    
    return b_df

# --- HELPER FOR TAB 2 SPEED ---
@st.cache_data(show_spinner=False)
def prep_dashboard_data(_df, _model, feats, perf_mode, d_start, d_end, p_min, p_max):
    mask = (_df['7:30AM Price'] >= p_min) & (_df['7:30AM Price'] <= p_max)
    if d_start and d_end:
        mask &= (_df['Date_DT'].dt.date >= d_start) & (_df['Date_DT'].dt.date <= d_end)
        
    res_data = _df[mask & (_df['Fin Pos'] > 0)].copy()
    if res_data.empty: return res_data
    
    if perf_mode == "Live":
        res_data['AI_R'] = pd.to_numeric(res_data.get('Rank', 0), errors='coerce')
    else:
        res_data['AI_S'] = _model.predict_proba(res_data[feats].fillna(0))[:, 1]
        res_data['AI_R'] = res_data.groupby(['Date_Key', 'Time', 'Course'])['AI_S'].rank(ascending=False, method='first')
    return res_data


# --- 4. CSS ---
st.markdown('<style>'
    '.block-container { padding-top: 3.5rem !important; }'
    '.scrollable-table { width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch; margin-bottom: 10px; border-radius: 4px; }'
    '.k2-table { border-collapse: collapse !important; width: 100% !important; min-width: 800px; table-layout: fixed !important; margin-bottom: 0px !important; }'
    '.k2-table th, .k2-table td { border: 1px solid #444 !important; padding: 3px 4px !important; font-size: 12.5px !important; white-space: nowrap !important; overflow: hidden !important; text-overflow: ellipsis !important; }'
    '.k2-table td.r1 { background-color: #2e7d32 !important; color: white !important; font-weight: bold !important; }'
    '.k2-table td.r2 { background-color: #fbc02d !important; color: black !important; font-weight: bold !important; }'
    '.k2-table td.r3 { background-color: #1976d2 !important; color: white !important; font-weight: bold !important; }'
    '.mauve-row td { background-color: #f3e5f5 !important; color: black !important; }'
    '.k2-table tr:hover td { background-color: #aec6cf !important; color: black !important; }'
    '.k2-table thead th { background-color: #000 !important; color: white !important; text-transform: uppercase; letter-spacing: 0.5px; }'
    '.left-head { text-align: left !important; padding-left: 10px !important; }'
    '.left-text { text-align: left !important; padding-left: 10px !important; }'
    '.center-text { text-align: center !important; }'
    '.pos-val { color: #2e7d32 !important; font-weight: bold !important; }'
    '.neg-val { color: #d32f2f !important; font-weight: bold !important; }'
'</style>', unsafe_allow_html=True)


# --- 5. EXECUTION & HEADER ---
model, feats, shadow_model, shadow_feats, cal_model, df_hist, df_live, df_today, last_live_date, first_res_date, df_all = load_all_data()

# --- CRITICAL FIX: Ensure app doesn't crash if data is missing ---
if model is None:
    st.error("🚨 Critical Error: Could not load the DailyAIResults.zip data file. Please ensure it is uploaded.")
    st.stop()

if 'expanded_races' not in st.session_state: st.session_state.expanded_races = set()

logo_b64 = ""
if os.path.exists("BOTManLogo.png"):
    with open("BOTManLogo.png", "rb") as f: logo_b64 = base64.b64encode(f.read()).decode()
logo_html = '<img src="data:image/png;base64,' + logo_b64 + '" height="55">' if logo_b64 else "BOTMan"

h_col1, h_col2 = st.columns([4.8, 2.0]) 
with h_col1:
    res_str = last_live_date.strftime('%d %b %Y').upper() if last_live_date else "08 MAR 2026"
    header_box = '<div style="display:flex; align-items:center; gap:20px; background-color:#1a3a5f; padding:15px; border-radius:10px; color:white; width: 100%; box-sizing: border-box;">' + logo_html + '<div>'
    header_box += '<div style="font-size:24px; font-weight:bold;">BOTMan Betting Systems</div>'
    header_box += '<div style="margin-top:5px;"><span style="background:#2e7d32; color:white; padding:2px 8px; border-radius:10px; font-size:12px;">✅ LIVE RESULTS TO ' + res_str + '</span></div>'
    header_box += '</div></div>'
    st.markdown(header_box, unsafe_allow_html=True)

with h_col2:
    if st.session_state.get("is_admin"):
        st.markdown('<div style="margin-top:0px;"></div>', unsafe_allow_html=True) 
        
        c_fast, c_slow = st.columns(2)
        with c_fast:
            if st.button("⚡ Daily Refresh", help="Instantly load new daily runners/systems", use_container_width=True):
                # Flush the daily access logs
                if os.path.exists("login_history.csv"):
                    os.remove("login_history.csv")
                
                st.cache_resource.clear()
                st.cache_data.clear()
                st.rerun()
        with c_slow:
            if st.button("🧠 Retrain AI", help="Rebuild AI brain after uploading new results (~5m)", use_container_width=True):
                if os.path.exists("botman_models.pkl"): 
                    os.remove("botman_models.pkl")
                st.cache_resource.clear()
                st.cache_data.clear()
                st.rerun()
        
        btn_label = "🔙 Return to Dashboard" if st.session_state.get("show_admin_insights") else "🔍 Admin Insights"
        if st.button(btn_label, key="admin_toggle_btn", use_container_width=True):
            st.session_state.show_admin_insights = not st.session_state.get("show_admin_insights", False)
            st.rerun()

# -------------------------------------------------------------------------
# VIEW CONTROLLER: Either show Admin Insights OR the Normal Sidebar Menu
# -------------------------------------------------------------------------
if st.session_state.get("is_admin") and st.session_state.get("show_admin_insights"):
    # --- ADMIN INSIGHTS VIEW ---
    st.header("🔍 Admin Data Insights")
    
    # --- NEW: Keeps the sidebar alive in Admin mode so you aren't trapped ---
    with st.sidebar:
        st.markdown("### ⚙️ Admin Mode")
        st.info("You are currently viewing the Admin Insights panel.")
        if st.button("🔙 Return to Dashboard", use_container_width=True):
            st.session_state.show_admin_insights = False
            st.rerun()
    
# --- NEW: LOGIN LOG VIEWER ---
    with st.expander("📋 View Daily App Access Logs", expanded=False):
        if os.path.exists("login_history.csv"):
            log_df = pd.read_csv("login_history.csv", names=["Date & Time", "User Type", "Session ID"])
            st.dataframe(log_df.sort_values(by="Date & Time", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("No login history recorded today. Logs clear automatically on Daily Refresh.")
    st.markdown("---")
    
    # --- PHASE 1: PREDICTION VAULT GENERATOR ---
    st.markdown("### 💾 Prediction Vault Management")
    st.info("Generates the permanent historical prediction file to prevent past results from changing when the AI retrains.")
    
    if st.button("Freeze Historical Predictions (Build Vault)", type="primary", use_container_width=True):
        with st.spinner("Processing 2 years of history... This may take a minute."):
            try:
                history_df = df_all[df_all['Fin Pos'] > 0].copy()
                vault_df = prep_system_builder_data(history_df, model, feats, shadow_model, shadow_feats, cal_model)
                
                # Strip it down to just the IDs and the Double-Brain AI Opinions
                vault_cols = ['Date', 'Time', 'Course', 'Horse', 'ML_Prob', 'Rank', 'Value Price', 'True_AI_Prob', 'Cal_Value_Price']
                available_cols = [c for c in vault_cols if c in vault_df.columns]
                final_vault = vault_df[available_cols]
                
                # Generate a compressed ZIP file to bypass upload limits
                import io
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    zip_file.writestr("BOTMan_Prediction_Vault.csv", final_vault.to_csv(index=False).encode('utf-8'))
                
                st.success(f"Vault generated successfully! ({len(final_vault)} records frozen)")
                st.download_button(
                    label="📥 Download Compressed Vault (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="BOTMan_Prediction_Vault.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error building vault: {e}")
                
    st.markdown("---")
            
    st.markdown("### Multi-Factor Analysis")
    st.markdown("Combine multiple data elements to discover highly profitable 'Golden Rules' hidden in your historical data.")
    
    if df_all is not None and not df_all.empty:
        ins_df = prep_system_builder_data(df_all, model, feats, shadow_model, shadow_feats, cal_model)
        
        i_col1, i_col2, i_col3 = st.columns([1.5, 1.5, 1])
        with i_col1:
            race_types_avail = ["All"] + sorted([str(x) for x in ins_df['Race Type'].dropna().unique() if str(x).strip()])
            race_filter = st.selectbox("Analyze Race Type:", race_types_avail)
        with i_col2:
            target_metric = st.selectbox(
                "Sort Results By:", 
                ["Logical Grouping (By Factor)", "Win P/L", "Win ROI (%)", "Win S/R (%)", "Place P/L", "Place ROI (%)", "Place S/R (%)"], 
                index=0
            )
        with i_col3:
            min_bets = st.number_input("Minimum Bets (Sample Size):", min_value=5, max_value=2000, value=25, step=5,
                                       help="Increase this to filter out wild outliers and find consistent trends.")
        
        analysis_cols = ['Rank', 'Comb. Rank', 'Speed Rank', 'Race Rank', 'No. of Top', 'Primary Rank', 'Form Rank', 'Class', 'Class Move', 'PRB Rank', 'Trainer PRB Rank', 'Jockey PRB Rank', 'MSAI Rank', 'Price Bracket', 'Pure Rank']
        avail_cols = [c for c in analysis_cols if c in ins_df.columns]
        
        selected_factors = st.multiselect("Select Factors to Combine (Choose 1 to 4):", avail_cols, default=['No. of Top', 'Speed Rank'])
        
        st.markdown("<div style='margin-top: 10px; margin-bottom: 5px; font-weight: bold; color: #1a3a5f;'>🎯 Filter by Objective Targets:</div>", unsafe_allow_html=True)
        o_col1, o_col2 = st.columns(2)
        with o_col1:
            min_sr = st.number_input("Minimum Win Strike Rate (%):", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
        with o_col2:
            min_roi = st.number_input("Minimum Win ROI (%):", min_value=-100.0, max_value=1000.0, value=0.0, step=5.0)
            
        if race_filter != "All":
            ins_df = ins_df[ins_df['Race Type'] == race_filter]
        
        st.markdown("---")
        
        if not selected_factors:
            st.info("💡 Leave the factors box empty and use the Auto-Discover Engine to find winning combinations.")
            
            c_btn, c_depth = st.columns([1, 1])
            with c_depth:
                search_depth = st.radio("Search Depth:", [1, 2], index=1, horizontal=True, help="1 = Single factors (e.g. Speed Rank 1). 2 = Paired factors (e.g. Speed Rank 1 + PRB Rank 2).")
            
            with c_btn:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🚀 Auto-Discover Golden Rules", type="primary", use_container_width=True):
                    import itertools
                    with st.spinner(f"Mining database for HIGH-VOLUME robust systems hitting {min_sr}% S/R and {min_roi}% ROI..."):
                        discovered_systems = []
                        disc_df = ins_df.copy()
                        smart_cols = []
                        
                        core_ranks = [c for c in ['Speed Rank', 'Form Rank', 'MSAI Rank', 'PRB Rank', 'Primary Rank', 'Pure Rank'] if c in disc_df.columns]
                        if core_ranks:
                            disc_df['Ratings in Top 3'] = (disc_df[core_ranks] <= 3).sum(axis=1).astype(str) + f" of {len(core_ranks)}"
                            smart_cols.append('Ratings in Top 3')
                            
                        allowed_ranks = ['Speed Rank', 'Primary Rank', 'Pure Rank', 'MSAI Rank']
                        for c in allowed_ranks:
                            if c in disc_df.columns:
                                disc_df[c] = pd.to_numeric(disc_df[c], errors='coerce')
                                disc_df[f"{c} Tier"] = pd.cut(disc_df[c], bins=[0, 1, 3, 999], labels=["Rank 1", "Ranks 2-3", "Garbage"])
                                smart_cols.append(f"{c} Tier")
                                
                        if 'Race Type' in disc_df.columns: smart_cols.append('Race Type')
                        if 'H/Cap' in disc_df.columns: smart_cols.append('H/Cap')
                        
                        combos = []
                        for r in range(1, search_depth + 1):
                            combos.extend(list(itertools.combinations(smart_cols, r)))
                            
                        progress_bar = st.progress(0)
                        total_combos = len(combos)
                        safe_min_bets = max(min_bets, 50) 
                        
                        for i, combo in enumerate(combos):
                            factors = list(combo)
                            grp = disc_df.groupby(factors, observed=False).agg(
                                Bets=('Horse', 'count'), Wins=('Is_Win', 'sum'), Profit=('Win P/L <2%', 'sum')
                            ).reset_index()
                            
                            grp = grp[grp['Bets'] >= safe_min_bets]
                            for f in factors:
                                if "Tier" in f:
                                    grp = grp[grp[f] != "Garbage"]
                            
                            if not grp.empty:
                                grp['Strike Rate (%)'] = (grp['Wins'] / grp['Bets']) * 100
                                grp['Win ROI (%)'] = (grp['Profit'] / grp['Bets']) * 100
                                winners = grp[(grp['Strike Rate (%)'] >= min_sr) & (grp['Win ROI (%)'] >= min_roi)].copy()
                                
                                for _, w in winners.iterrows():
                                    rule_name = " + ".join([f"{f.replace(' Tier', '')}: {w[f]}" for f in factors])
                                    discovered_systems.append({
                                        "Winning Rule": rule_name,
                                        "Bets": int(w['Bets']),
                                        "Wins": int(w['Wins']),
                                        "S/R (%)": round(w['Strike Rate (%)'], 1),
                                        "ROI (%)": round(w['Win ROI (%)'], 1),
                                        "Win P/L": round(w['Profit'], 2)
                                    })
                            progress_bar.progress((i + 1) / total_combos)
                            
                        progress_bar.empty()
                        
                        if discovered_systems:
                            st.success(f"🔥 Found {len(discovered_systems)} robust rules (All enforced to have Min {safe_min_bets} bets)!")
                            res_df = pd.DataFrame(discovered_systems).drop_duplicates(subset=["Winning Rule"])
                            sort_map = {"Win P/L": "Win P/L", "Win ROI (%)": "ROI (%)", "Win S/R (%)": "S/R (%)"}
                            sort_col = sort_map.get(target_metric, 'Win P/L')
                            res_df = res_df.sort_values(sort_col, ascending=False).head(100)
                            st.dataframe(res_df, use_container_width=True, hide_index=True)
                        else:
                            st.warning(f"No robust combinations found. (Filtered for a strict minimum of {safe_min_bets} bets to prevent random luck). Try lowering targets.")
        else:
            st.markdown(f"### 🏆 System Analysis for {race_filter} Races")
            
            grp = ins_df.groupby(selected_factors, observed=False).agg(
                Bets=('Horse', 'count'),
                Wins=('Is_Win', 'sum'),
                Profit=('Win P/L <2%', 'sum'),
                Places=('Is_Place', 'sum'),
                Place_Profit=('Place P/L <2%', 'sum')
            ).reset_index()
            
            grp = grp[grp['Bets'] >= min_bets] 
            
            if 'Price Bracket' in selected_factors:
                grp = grp[grp['Price Bracket'] != 'Unknown']
            
            if not grp.empty:
                grp['Strike Rate (%)'] = (grp['Wins'] / grp['Bets']) * 100
                grp['Win ROI (%)'] = (grp['Profit'] / grp['Bets']) * 100
                grp['Place SR (%)'] = (grp['Places'] / grp['Bets']) * 100
                grp['Place ROI (%)'] = (grp['Place_Profit'] / grp['Bets']) * 100
                grp['Total P/L'] = grp['Profit'] + grp['Place_Profit']
                
                grp = grp[(grp['Strike Rate (%)'] >= min_sr) & (grp['Win ROI (%)'] >= min_roi)]
                
                if grp.empty:
                    st.info(f"No factor combinations met your strict targets (Min {min_sr}% S/R and {min_roi}% ROI). Try lowering your objectives.")
                else:
                    if target_metric == "Logical Grouping (By Factor)":
                        ascending_sorts = []
                        for factor in selected_factors:
                            if factor == 'No. of Top':
                                ascending_sorts.append(False) 
                            else:
                                ascending_sorts.append(True)  
                        grp = grp.sort_values(by=selected_factors, ascending=ascending_sorts)
                    else:
                        sort_map = {
                            "Win P/L": "Profit",
                            "Win ROI (%)": "Win ROI (%)",
                            "Win S/R (%)": "Strike Rate (%)",
                            "Place P/L": "Place_Profit",
                            "Place ROI (%)": "Place ROI (%)",
                            "Place S/R (%)": "Place SR (%)"
                        }
                        sort_col = sort_map.get(target_metric, 'Profit')
                        grp = grp.sort_values(by=sort_col, ascending=False).head(100)
                    
                    html_table = """
                    <style>
                        .builder-table { border-collapse: collapse; width: 100%; min-width: 900px; font-size: 14px; font-family: sans-serif; }
                        .builder-table th, .builder-table td { border: 1px solid #ccc; padding: 4px; text-align: center; white-space: nowrap; }
                        .left-align { text-align: left !important; padding-left: 8px !important; }
                        .divider { border-left: 3px solid #1a3a5f !important; }
                    </style>
                    <div class="scrollable-table">
                    <table class="builder-table">
                        <thead><tr style="background-color: #1a3a5f; color: white;">
                    """
                    
                    for factor in selected_factors:
                        html_table += f"<th>{factor}</th>"
                        
                    html_table += """
                                <th class="divider">Total Bets</th><th>Wins</th><th>Win P/L</th><th>Win S/R</th><th>Win ROI</th>
                                <th class="divider">Places</th><th>Plc P/L</th><th>Plc S/R</th><th>Total P/L</th>
                            </tr></thead><tbody>
                    """
                    
                    for _, r in grp.iterrows():
                        html_table += "<tr>"
                        
                        for factor in selected_factors:
                            val = r[factor]
                            if isinstance(val, float):
                                val = int(val) if val.is_integer() else f"{val:.1f}"
                            html_table += f"<td style='color:#1a3a5f; font-weight:bold;'>{val}</td>"
                            
                        p_color = "#2e7d32" if r['Profit'] > 0 else "#d32f2f"
                        r_color = "#2e7d32" if r['Win ROI (%)'] > 0 else "#d32f2f"
                        pp_color = "#2e7d32" if r['Place_Profit'] > 0 else "#d32f2f"
                        t_color = "#2e7d32" if r['Total P/L'] > 0 else "#d32f2f"
                            
                        html_table += f"""
                            <td class="divider">{int(r['Bets'])}</td>
                            <td>{int(r['Wins'])}</td>
                            <td style="color:{p_color}; font-weight:bold;">£{r['Profit']:.2f}</td>
                            <td>{r['Strike Rate (%)']:.1f}%</td>
                            <td style="color:{r_color}; font-weight:bold;">{r['Win ROI (%)']:.1f}%</td>
                            <td class="divider">{int(r['Places'])}</td>
                            <td style="color:{pp_color}; font-weight:bold;">£{r['Place_Profit']:.2f}</td>
                            <td>{r['Place SR (%)']:.1f}%</td>
                            <td style="color:{t_color}; font-weight:bold;">£{r['Total P/L']:.2f}</td>
                        </tr>"""
                    html_table += "</tbody></table></div>"
                    st.markdown(html_table, unsafe_allow_html=True)
            else:
                st.info(f"No combinations found with at least {min_bets} bets.")
    else:
        st.warning("No data available.")
else:
    # --- NORMAL DASHBOARD VIEW ---
    with st.sidebar:
        st.markdown("### 🧭 Main Menu")
        app_mode = st.radio("Navigate to:", [
            "📅 Daily Predictions", 
            "📊 AI Top 2 Results", 
            "🧠 General Systems", 
            "🛠️ System Builder", 
            "🏇 Race Analysis"
        ])

# --- Page 1: Daily Predictions ---
    if app_mode == "📅 Daily Predictions":
        st.header("📅 Daily Top 2 Predictions")
        
        if df_today is not None and not df_today.empty:
            if 'ML_Prob' in df_today.columns:
                df_p = df_today.copy()
                df_p.columns = df_p.columns.str.strip()
                
                df_p['Rank'] = df_p.groupby(['Date', 'Time', 'Course'])['ML_Prob'].rank(ascending=False, method='min')
                
                if 'No. of Top' in df_p.columns:
                    df_p['No. of Top'] = pd.to_numeric(df_p['No. of Top'], errors='coerce').fillna(0)
                    df_p['Max_Top'] = df_p.groupby(['Date', 'Time', 'Course'])['No. of Top'].transform('max')
                    df_p['isM'] = (df_p['No. of Top'] == df_p['Max_Top']) & (df_p['No. of Top'] > 0)
                else:
                    df_p['isM'] = False
                
                df_p = df_p.sort_values(by=['Date', 'Time', 'Course', 'Rank'])
                ideal_csv_cols = ['Date', 'Time', 'Course', 'Horse', '7:30AM Price', 'ML_Prob', 'Rank', 'No. of Top']
                existing_csv_cols = [c for c in ideal_csv_cols if c in df_p.columns]
                
                csv_out_top2 = df_p[df_p['Rank'] <= 2][existing_csv_cols].copy()
                csv_out_full = df_p[existing_csv_cols].copy()
                
                timestamp = datetime.now().strftime('%d%m%y_%H%M%S')
                file_top2 = f"BOTMan_Top2_AIPredictions_{timestamp}.csv"
                file_full = f"BOTMan_Full_AIPredictions_{timestamp}.csv"
                
                col_dl1, col_dl2, col_spacer, col_col = st.columns([1.2, 1.2, 2.1, 0.5])
                
                with col_dl1:
                    st.download_button("📥 Download Top 2 (CSV)", csv_out_top2.to_csv(index=False).encode('utf-8'), file_top2)
                
                with col_dl2:
                    st.download_button("📥 Download All (CSV)", csv_out_full.to_csv(index=False).encode('utf-8'), file_full)
                
                with col_col:
                    if st.button("Collapse All"): 
                        st.session_state.expanded_races = set()
                        st.rerun()

                w = ["10%", "10%", "12%", "31%", "12%", "12%", "8%", "5%"]
                
                h_col1, h_col2 = st.columns([19, 1], gap="small")
                with h_col1:
                    header = '<div class="scrollable-table"><table class="k2-table"><thead><tr>'
                    header += f'<th style="width:{w[0]};" class="left-head">Date</th><th style="width:{w[1]};" class="left-head">Time</th>'
                    header += f'<th style="width:{w[2]};" class="left-head">Course</th><th style="width:{w[3]};" class="left-head">Horse</th>'
                    header += f'<th style="width:{w[4]};" class="left-head">Price</th><th style="width:{w[5]};" class="left-head">AI Prob</th>'
                    header += f'<th style="width:{w[6]};" class="left-head">Rank</th><th style="width:{w[7]};" class="left-head">Tops</th>'
                    header += '</tr></thead></table></div>'
                    st.markdown(header, unsafe_allow_html=True)

                for (d, t, c), group in df_p.groupby(['Date', 'Time', 'Course'], sort=False):
                    race_id = str(d) + " " + str(t) + " " + str(c)
                    is_expanded = race_id in st.session_state.expanded_races
                    rows = group if is_expanded else group[group['Rank'] <= 2]
                    
                    st.markdown('<div style="margin-top:2px;"></div>', unsafe_allow_html=True)
                    
                    t_col, b_col = st.columns([19, 1], gap="small")
                    with t_col:
                        html = '<div class="scrollable-table"><table class="k2-table"><tbody>'
                        for _, r in rows.iterrows():
                            row_cls = "mauve-row" if r['isM'] else ""
                            rv = int(r['Rank'])
                            r_cls = "r1" if rv==1 else "r2" if rv==2 else "r3" if rv==3 else ""
                            html += f'<tr class="{row_cls}">'
                            html += f'<td style="width:{w[0]};" class="center-text">{r["Date"]}</td>'
                            html += f'<td style="width:{w[1]};" class="center-text">{r["Time"]}</td>'
                            html += f'<td style="width:{w[2]};" class="left-text">{r["Course"]}</td>'
                            html += f'<td style="width:{w[3]};" class="left-text"><b>{r["Horse"]}</b></td>'
                            html += f'<td style="width:{w[4]};" class="center-text">{round(r["7:30AM Price"], 2)}</td>'
                            html += f'<td style="width:{w[5]};" class="center-text">{round(r["ML_Prob"], 4)}</td>'
                            html += f'<td style="width:{w[6]};" class="{r_cls} center-text">{rv}</td>'
                            html += f'<td style="width:{w[7]};" class="center-text">{int(r["No. of Top"])}</td>'
                            html += '</tr>'
                        st.markdown(html + '</tbody></table></div>', unsafe_allow_html=True)
                    with b_col:
                        if st.button("-" if is_expanded else "+", key="btn_"+race_id):
                            if is_expanded: st.session_state.expanded_races.remove(race_id)
                            else: st.session_state.expanded_races.add(race_id)
                            st.rerun()

# --- Page 2: Dashboard ---
    elif app_mode == "📊 AI Top 2 Results":
        if "perf_mode" not in st.session_state: st.session_state.perf_mode = "Live"
        st.markdown('<div class="filter-area">', unsafe_allow_html=True)
        cb1, cb2, cd = st.columns([1, 1, 2])
        if cb1.button("Recent Results (Live)", type="primary" if st.session_state.perf_mode == "Live" else "secondary"): 
            st.session_state.perf_mode = "Live"; st.rerun()
        if cb2.button("Historical Data", type="primary" if st.session_state.perf_mode == "Legacy" else "secondary"): 
            st.session_state.perf_mode = "Legacy"; st.rerun()
        
        ytd_start = datetime(2026, 3, 9).date()
        if st.session_state.perf_mode == "Live":
            df_s = df_live
            f_end = last_live_date.date() if last_live_date else datetime.now().date()
            d_range = cd.date_input("Live Range (Since 9th March)", [ytd_start, f_end], min_value=ytd_start)
        else:
            df_s = df_hist
            f_start = first_res_date.date() if first_res_date else datetime(2024,1,1)
            d_range = cd.date_input("Historical Range (To 8th March)", [f_start, datetime(2026, 3, 8).date()], max_value=datetime(2026, 3, 8).date())
        
        price_options = {
            "All": (0.0, 1000.0), "Up to 5": (0.0, 5.0), "5 to 10": (5.01, 10.0),
            "10 to 15": (10.01, 15.0), "15 to 25": (15.01, 25.0), "25 to 50": (25.01, 50.0),
            "50 to 100": (50.01, 100.0), "> 100": (100.01, 1000.0)
        }
        sel_range = st.radio("Price Range Quick-Select:", list(price_options.keys()), index=0, horizontal=True)
        start_p, end_p = price_options[sel_range]
        p_range = st.slider("Fine-Tune Price Filter", 0.0, 1000.0, (float(start_p), float(end_p)))
        st.markdown('</div>', unsafe_allow_html=True)
        
        if df_s is not None and not df_s.empty:
            d_start = d_range[0] if len(d_range) > 0 else None
            d_end = d_range[1] if len(d_range) == 2 else d_start
            
            master_tab2_df = prep_dashboard_data(df_s, model, feats, st.session_state.perf_mode, d_start, d_end, p_range[0], p_range[1])
            audit_totals = {'Win': 0.0, 'Place': 0.0}
            
            def render_pick_card(label, data, is_main_cat=False):
                if data.empty: return
                p1_count = len(data[data['AI_R'] == 1])
                p2_count = len(data[data['AI_R'] == 2])
                total_runs = p1_count + p2_count
                st.markdown(f'''<div style="border: 1px solid #444; background-color: #f9f9f9; padding: 10px; border-radius: 4px; margin-bottom: 15px;"><div style="border-left: 6px solid #1a3a5f; padding-left: 10px; margin-bottom: 10px; font-weight: bold; font-size: 14px;">{label} <span style="font-weight: normal; color: #666;">(Total: {total_runs})</span></div>''', unsafe_allow_html=True)
                cols = st.columns(2)
                for i in range(1, 3):
                    pick = data[data['AI_R'] == i]
                    pr = len(pick)
                    with cols[i-1]:
                        if pr > 0:
                            wpl = float(pick['Win P/L <2%'].sum())
                            ppl = float(pick['Place P/L <2%'].sum())
                            wsr = (len(pick[pick['Fin Pos']==1])/pr)*100
                            psr = (len(pick[pick['Fin Pos'] <= 3])/pr)*100
                            wroi, proi = (wpl/pr)*100, (ppl/pr)*100
                            if is_main_cat: audit_totals['Win'] += wpl; audit_totals['Place'] += ppl
                            wpl_cls = "pos-val" if wpl >= 0 else "neg-val"
                            ppl_cls = "pos-val" if ppl >= 0 else "neg-val"
                            wroi_cls = "pos-val" if wroi >= 0 else "neg-val"
                            proi_cls = "pos-val" if proi >= 0 else "neg-val"
                            bg, tx = ('#2e7d32', 'white') if i==1 else ('#fbc02d', 'black')
                            
                            c_box = f'''<div class="pick-box" style="border: 1px solid #ccc; background: white;"><div style="background:{bg}; color:{tx}; text-align:center; font-weight:bold; font-size:12px; padding:3px;">PICK {i} ({pr})</div><div style="padding:8px; font-size:11px; line-height:1.6;"><b>Win P&L:</b> <span class="{wpl_cls}">{round(wpl, 2)}</span> | <b>S/R%:</b> {round(wsr, 1)} | <b>ROI%:</b> <span class="{wroi_cls}">{round(wroi, 1)}%</span><br><b>Place P&L:</b> <span class="{ppl_cls}">{round(ppl, 2)}</span> | <b>S/R%:</b> {round(psr, 1)} | <b>ROI%:</b> <span class="{proi_cls}">{round(proi, 1)}%</span></div></div>'''
                            st.markdown(c_box, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div style="background:#1a3a5f; color:white; padding:8px 15px; font-weight:bold; border-radius:4px; margin-bottom:10px;">TOTAL SYSTEM BREAKDOWN</div>', unsafe_allow_html=True)
            render_pick_card("TOTAL SYSTEM", master_tab2_df, is_main_cat=True)
            
            sc1, sc2 = st.columns(2)
            with sc1: render_pick_card("TOTAL NON-HANDICAP", master_tab2_df[master_tab2_df['H/Cap'].astype(str).str.strip() == 'N'])
            with sc2: render_pick_card("TOTAL HANDICAP", master_tab2_df[master_tab2_df['H/Cap'].astype(str).str.strip() == 'Y'])
            
            for rt in ['A/W', 'Chase', 'Hurdle', 'Turf']:
                st.markdown(f'<div style="background:#1a3a5f; color:white; padding:8px 15px; font-weight:bold; border-radius:4px; margin-top:20px; margin-bottom:10px;">{rt} CATEGORY BREAKDOWN</div>', unsafe_allow_html=True)
                render_pick_card(rt+" Aggregated", master_tab2_df[master_tab2_df['Race Type'].astype(str).str.strip() == rt])
                sc1, sc2 = st.columns(2)
                with sc1: render_pick_card(rt+" Non-Handicap", master_tab2_df[(master_tab2_df['Race Type'].astype(str).str.strip() == rt) & (master_tab2_df['H/Cap'].astype(str).str.strip() == 'N')])
                with sc2: render_pick_card(rt+" Handicap", master_tab2_df[(master_tab2_df['Race Type'].astype(str).str.strip() == rt) & (master_tab2_df['H/Cap'].astype(str).str.strip() == 'Y')])

            if st.session_state.perf_mode == "Live":
                st.markdown('<div style="background:#1a3a5f; color:white; padding:6px 15px; font-weight:bold; border-radius:4px; margin-top:20px;">🏆 LIVE TRACK PERFORMANCE RANKINGS</div>', unsafe_allow_html=True)
                track_stats = []
                for course in df_s['Course'].unique():
                    c_data = df_s[(df_s['Course'] == course) & (df_s['Fin Pos'] > 0)]
                    c_runs = len(c_data)
                    if c_runs >= 2:
                        c_win = float(c_data['Win P/L <2%'].sum())
                        c_roi = (c_win / c_runs) * 100
                        track_stats.append({'Course': course, 'Picks': c_runs, 'Win P/L': c_win, 'ROI%': c_roi})
                if track_stats:
                    top_tracks = pd.DataFrame(track_stats).sort_values('ROI%', ascending=False).head(5)
                    st.table(top_tracks.set_index('Course'))
# --- Page 3: General Systems Dashboard ---
    elif app_mode == "🧠 General Systems":
        st.header("🧠 General Systems")
        
        smart_view = st.radio("Select View:", ["📅 Today's Qualifiers", "📊 Live Performance (Master file)"], horizontal=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        if smart_view == "📅 Today's Qualifiers":
            s_col, p_col = st.columns(2)
            with s_col:
                sort_pref = st.radio("Sort Qualifiers By:", ["System Name (Morning Review)", "Time (Live Racing)"], horizontal=True)
            with p_col:
                if st.session_state.get("is_admin"):
                    pool_choice = st.radio("System Pool (Admin Only):", ["Public", "Admin Secret", "Combined"], horizontal=True)
                else:
                    pool_choice = "Public"
            
            st.markdown("<br>", unsafe_allow_html=True)
            try:
                if df_today is not None and not df_today.empty:
                    if 'ML_Prob' not in df_today.columns:
                        st.warning("Data Check: 'ML_Prob' could not be calculated. Please check your source CSV for missing columns.")
                    else:
                        all_today_picks = []
                        t_df = prep_system_builder_data(df_today, model, feats, shadow_model, shadow_feats, cal_model, is_live_today=True)

                        saved_systems = {}
                        if pool_choice in ["Public", "Combined"] and os.path.exists("BOTMan_user_systems.json"):
                            with open("BOTMan_user_systems.json", "r") as f:
                                try: saved_systems.update(json.load(f))
                                except: pass
                                
                        if pool_choice in ["Admin Secret", "Combined"] and os.path.exists("BOTMan_admin_systems.json"):
                            with open("BOTMan_admin_systems.json", "r") as f:
                                try: saved_systems.update(json.load(f))
                                except: pass

                        if saved_systems:
                            for s_name, s_data in saved_systems.items():
                                s_mask = (
                                    t_df['Race Type'].isin(s_data.get('race_types', [])) &
                                    t_df['H/Cap'].isin(s_data.get('hcap_types', [])) &
                                    (t_df['7:30AM Price'] >= s_data.get('price_min', 0.0)) &
                                    (t_df['7:30AM Price'] <= s_data.get('price_max', 1000.0)) &
                                    (t_df['Prob Gap'] >= s_data.get('min_prob_gap', -100.0)) &
                                    (t_df['Value_Edge_Perc'] >= s_data.get('min_edge_perc', -100.0))
                                )
                                
                                if s_data.get('rank_1_only', False): s_mask &= (t_df['Rank'] == 1)
                                
                                months = s_data.get('months', ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
                                if len(months) < 12:
                                    month_map = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
                                    sel_m_nums = [month_map[m] for m in months]
                                    s_mask &= t_df['Date_DT'].dt.month.isin(sel_m_nums)

                                vf = s_data.get('value_filter', "Off")
                                if vf in ["Value vs 7:30AM Price", "Value vs BSP", "Original AI vs 7:30AM", "Original AI vs BSP", "AI Value vs 7:30AM", "AI Value vs BSP"]: 
                                    s_mask &= (t_df['7:30AM Price'] > t_df['Value Price'])
                                elif vf in ["Calibrated AI (Leashed) vs 7:30AM", "Calibrated AI (Leashed) vs BSP"]:
                                    s_mask &= (t_df['7:30AM Price'] > t_df['Cal_Value_Price'])
                                elif vf in ["My Value vs 7:30AM", "My Value vs BSP"]: 
                                    s_mask &= (t_df['7:30AM Price'] > t_df['User Value'])

                                rnrs = s_data.get('rnrs', [])
                                r_m = pd.Series(False, index=t_df.index)
                                if "2-7" in rnrs: r_m |= (t_df['No. of Rnrs'] >= 2) & (t_df['No. of Rnrs'] <= 7)
                                if "8-12" in rnrs: r_m |= (t_df['No. of Rnrs'] >= 8) & (t_df['No. of Rnrs'] <= 12)
                                if "13-16" in rnrs: r_m |= (t_df['No. of Rnrs'] >= 13) & (t_df['No. of Rnrs'] <= 16)
                                if ">16" in rnrs: r_m |= (t_df['No. of Rnrs'] > 16)
                                if not r_m.any() and not rnrs: r_m = pd.Series(True, index=t_df.index)
                                s_mask &= r_m

                                if 'Class' in t_df.columns and s_data.get('classes'): s_mask &= t_df['Class'].isin(s_data['classes'])
                                if 'Class Move' in t_df.columns and s_data.get('cm'): s_mask &= t_df['Class Move'].isin(s_data['cm'])
                                if 'Age' in t_df.columns: s_mask &= (t_df['Age'] >= s_data.get('age_min', 1)) & (t_df['Age'] <= s_data.get('age_max', 20))

                                irish_setting = s_data.get('irish', "Any")
                                irish_col = 'Irish?' if 'Irish?' in t_df.columns else 'Irish' if 'Irish' in t_df.columns else None
                                if irish_col and irish_setting != "Any":
                                    t_irish_series = t_df[irish_col].astype(str).str.strip().str.upper()
                                    if irish_setting == "Y (Yes)": s_mask &= (t_irish_series == 'Y')
                                    elif irish_setting == "No (Blank)": s_mask &= (t_irish_series != 'Y')

                                if 'Sex' in t_df.columns and s_data.get('sex'): s_mask &= t_df['Sex'].astype(str).str.strip().str.lower().isin([s.lower() for s in s_data['sex']])
                                if 'Course' in t_df.columns and s_data.get('courses'): s_mask &= t_df['Course'].astype(str).str.strip().isin(s_data['courses'])

                                ranks = s_data.get('ranks', {})
                                for col_name, setting in ranks.items():
                                    if setting != "Any" and col_name in t_df.columns:
                                        num_col = pd.to_numeric(t_df[col_name], errors='coerce')
                                        if setting == "Rank 1": s_mask &= (num_col == 1)
                                        elif setting == "Top 2": s_mask &= (num_col <= 2)
                                        elif setting == "Top 3": s_mask &= (num_col <= 3)
                                        elif setting == "Top 4": s_mask &= (num_col <= 4)
                                        elif setting == "Top 5": s_mask &= (num_col <= 5)

                                sys_df = t_df[s_mask].copy()
                                if not sys_df.empty:
                                    sys_df['System Name'] = s_name
                                    all_today_picks.append(sys_df)

                        if all_today_picks:
                            final_df = pd.concat(all_today_picks, ignore_index=True)
                            
                            if pool_choice == "Public":
                                ideal_base_cols = ["Date", "Time", "Course", "Horse", "7:30AM Price", "ML_Prob", "Rank", "No. of Top", "System Name"]
                            else:
                                ideal_base_cols = ["Date", "Time", "Course", "Horse", "7:30AM Price", "ML_Prob", "Rank", "Primary Rank", "Pure Rank", "No. of Top", "System Name"]
                                
                            existing_cols = [c for c in ideal_base_cols if c in final_df.columns]
                            final_df = final_df[existing_cols]
                            
                            if sort_pref == "System Name (Morning Review)": final_df = final_df.sort_values(by=["System Name", "Date", "Time", "Course"], ascending=[True, True, True, True])
                            else: final_df = final_df.sort_values(by=["Date", "Time", "Course", "System Name"], ascending=[True, True, True, True])

                            unique_systems = final_df['System Name'].unique()
                            palette = ["#e8f4f8", "#f8e8e8", "#e8f8e8", "#f8f4e8", "#f4e8f8", "#e8f8f8"]
                            sys_color_map = {sys: palette[i % len(palette)] for i, sys in enumerate(unique_systems)}

                            csv_data = final_df.to_csv(index=False).encode('utf-8')
                            timestamp = datetime.now().strftime('%d%m%y_%H%M%S')
                            dl_label = "📥 Download Admin Picks to CSV" if pool_choice != "Public" else "📥 Download General Picks to CSV"
                            st.download_button(dl_label, csv_data, f"BOTMan_{pool_choice}_Systems_{timestamp}.csv", "text/csv", key="dl_smart")
                            st.write("")

                            html_table = """<style>.contiguous-table { border-collapse: collapse; width: 100%; min-width: 900px; font-size: 14px; font-family: sans-serif; } .contiguous-table th, .contiguous-table td { border: 1px solid #ccc; padding: 4px; text-align: left; white-space: nowrap; } .contiguous-table tr:hover { background-color: #0000FF !important; color: white !important; }</style><div class="scrollable-table"><table class="contiguous-table"><thead><tr>"""
                            for col in existing_cols: html_table += f"<th>{col}</th>"
                            html_table += "</tr></thead><tbody>"

                            for _, row in final_df.iterrows():
                                bg_color = sys_color_map.get(row.get("System Name"), "")
                                text_color = "black"
                                row_style = f"background-color: {bg_color}; color: {text_color};" if bg_color else ""
                                html_table += f"<tr style='{row_style}'>"
                                for col in existing_cols:
                                    val = row[col]
                                    if isinstance(val, float):
                                        if col == "ML_Prob": val = f"{val*100:.1f}%"
                                        elif col in ["Rank", "No. of Top", "Primary Rank", "Pure Rank"]: val = f"{int(val)}"
                                        else: val = f"{val:.2f}"
                                    html_table += f"<td>{val}</td>"
                                html_table += "</tr>"

                            html_table += "</tbody></table></div>"
                            st.markdown(html_table, unsafe_allow_html=True)
                        else: st.info(f"No systems selections found today for the '{pool_choice}' pool.")
                else: st.info("No data available for today's races.")

            except Exception as e: st.error(f"Error loading General Systems: {e}")
                
        else:
            st.markdown("### 📈 Live Performance (Master file)")
            
            if st.session_state.get("is_admin"):
                perf_file_choice = st.radio("Select Master File to Analyze:", ["Public (BOTManSystemsMaster.ods)", "Admin (BOTManAdminMaster.ods)"], horizontal=True)
                target_ods = "BOTManAdminMaster.ods" if "Admin" in perf_file_choice else "BOTManSystemsMaster.ods"
            else:
                target_ods = "BOTManSystemsMaster.ods"
                
            if os.path.exists(target_ods):
                try:
                    df_smart_master = pd.read_excel(target_ods, engine="odf")
                    df_smart_master.columns = df_smart_master.columns.str.strip()
                    if all(c in df_smart_master.columns for c in ['Date', 'Time', 'Course', 'Horse']) and df_all is not None:
                        sys_col_found = None
                        for col in df_smart_master.columns:
                            if col.lower() in ['system name', 'system', 'system_name', 'systems']:
                                sys_col_found = col
                                break
                        
                        def clean_d(x):
                            s = str(x).split('.')[0].strip()
                            return s[-6:] if len(s) > 6 else s
                            
                        # --- THE SURGICAL FIX FOR DATA TYPES ---
                        # Force ODS columns to be clean strings to match the database
                        df_smart_master['Date_Key'] = df_smart_master['Date'].apply(clean_d)
                        df_smart_master['Time'] = df_smart_master['Time'].astype(str).str.split('.').str[0].str.strip()
                        df_smart_master['Course'] = df_smart_master['Course'].astype(str).str.strip().str.title()
                        df_smart_master['Horse'] = df_smart_master['Horse'].astype(str).str.strip().str.title()
                        
                        # Use the Full Database (df_all) and force its types to match
                        db_clean = df_all.copy()
                        db_clean['Time'] = db_clean['Time'].astype(str).str.split('.').str[0].str.strip()
                        db_clean['Course'] = db_clean['Course'].astype(str).str.strip().str.title()
                        db_clean['Horse'] = db_clean['Horse'].astype(str).str.strip().str.title()

                        # Prevent column collisions (_x/_y) by only merging onto the Master ZIP data
                        keep_keys = ['Date_Key', 'Time', 'Course', 'Horse']
                        if sys_col_found: keep_keys.append(sys_col_found)
                        ods_keys_only = df_smart_master[keep_keys].copy()
                        
                        merged_smart = pd.merge(ods_keys_only, db_clean, on=['Date_Key', 'Time', 'Course', 'Horse'], how='inner')
                        # ---------------------------------------

                        merged_smart['Fin Pos'] = pd.to_numeric(merged_smart['Fin Pos'], errors='coerce')
                        merged_smart = merged_smart[merged_smart['Fin Pos'] > 0]
                        
                        if not merged_smart.empty:
                            # Run the Double-Brain math
                            merged_smart = prep_system_builder_data(merged_smart, model, feats, shadow_model, shadow_feats, cal_model, is_live_today=False, use_vault=True)
                            
                            if sys_col_found is None:
                                merged_smart['System Name'] = 'All Systems Combined'
                                sys_col_found = 'System Name'
                            else:
                                merged_smart['System Name'] = merged_smart[sys_col_found]
                                sys_col_found = 'System Name'

                            merged_smart['Win P/L <2%'] = pd.to_numeric(merged_smart['Win P/L <2%'], errors='coerce').fillna(0)
                            merged_smart['Place P/L <2%'] = pd.to_numeric(merged_smart['Place P/L <2%'], errors='coerce').fillna(0)
                            merged_smart['Is_Win'] = np.where(merged_smart['Win P/L <2%'] > 0, 1, 0)
                            merged_smart['Is_Place'] = np.where((merged_smart['Fin Pos'] >= 1) & (merged_smart['Fin Pos'] <= 3), 1, 0)
                            
                            merged_smart['Date_DT'] = pd.to_datetime(merged_smart['Date_Key'], format='%y%m%d', errors='coerce')
                            merged_smart['Month_Yr'] = merged_smart['Date_DT'].dt.strftime('%Y - %b')
                            current_month_str = datetime.now().strftime('%Y - %b')
                            
                            all_time = merged_smart.groupby(sys_col_found, observed=False).agg(
                                Bets=('Horse', 'count'), Wins=('Is_Win', 'sum'), Win_Profit=('Win P/L <2%', 'sum'), Places=('Is_Place', 'sum'), Place_Profit=('Place P/L <2%', 'sum')
                            ).reset_index()
                            all_time['Period'] = 'All Time'
                            
                            curr_month_df = merged_smart[merged_smart['Month_Yr'] == current_month_str]
                            if not curr_month_df.empty:
                                curr_month = curr_month_df.groupby(sys_col_found, observed=False).agg(
                                    Bets=('Horse', 'count'), Wins=('Is_Win', 'sum'), Win_Profit=('Win P/L <2%', 'sum'), Places=('Is_Place', 'sum'), Place_Profit=('Place P/L <2%', 'sum')
                                ).reset_index()
                                curr_month['Period'] = current_month_str
                            else:
                                curr_month = all_time.copy()
                                curr_month['Period'] = current_month_str
                                curr_month[['Bets', 'Wins', 'Win_Profit', 'Places', 'Place_Profit']] = 0

                            combined = pd.concat([all_time, curr_month], ignore_index=True)
                            combined['Strike Rate (%)'] = np.where(combined['Bets'] > 0, (combined['Wins'] / combined['Bets'] * 100), 0)
                            combined['Place SR (%)'] = np.where(combined['Bets'] > 0, (combined['Places'] / combined['Bets'] * 100), 0)
                            combined['Win ROI (%)'] = np.where(combined['Bets'] > 0, (combined['Win_Profit'] / combined['Bets'] * 100), 0)
                            combined['Total P/L'] = combined['Win_Profit'] + combined['Place_Profit']
                            
                            combined['SortKey'] = np.where(combined['Period'] == 'All Time', 1, 2)
                            combined = combined.sort_values(by=[sys_col_found, 'SortKey']).drop('SortKey', axis=1)

                            html_table = """<style>.builder-table { border-collapse: collapse; width: 100%; min-width: 1000px; font-size: 14px; font-family: sans-serif; margin-top: 15px; } .builder-table th, .builder-table td { border: 1px solid #ccc; padding: 6px; text-align: center; white-space: nowrap; } .builder-table tr:hover { background-color: #0000FF !important; color: white !important; } .left-align { text-align: left !important; padding-left: 8px !important; }</style><div class="scrollable-table"><table class="builder-table"><thead><tr style="background-color: #1a3a5f; color: white;"><th class="left-align">System Name</th><th class="left-align">Period</th><th>Bets</th><th>Wins</th><th>Win P/L</th><th>Win SR</th><th>Places</th><th>Plc P/L</th><th>Plc SR</th><th>Total P/L</th><th>DL</th></tr></thead><tbody>"""
                            
                            unique_sys = combined[sys_col_found].unique()
                            palette = ["#e8f4f8", "#f8e8e8", "#e8f8e8", "#f8f4e8", "#f4e8f8", "#e8f8f8"]
                            bg_colors = {sys: palette[i % len(palette)] for i, sys in enumerate(unique_sys)}

                            last_sys = None
                            for _, row in combined.iterrows():
                                if last_sys is not None and last_sys != row[sys_col_found]:
                                    html_table += '<tr><td colspan="11" style="border: none !important; background-color: white !important; height: 15px; padding: 0 !important;"></td></tr>'
                                last_sys = row[sys_col_found]
                                bg = bg_colors[row[sys_col_found]]
                                b_s = "<b>" if row['Period'] == 'All Time' else ""
                                b_e = "</b>" if row['Period'] == 'All Time' else ""
                                w_color = "#2e7d32" if row['Win_Profit'] > 0 else "#d32f2f" if row['Win_Profit'] < 0 else "black"
                                p_color = "#2e7d32" if row['Place_Profit'] > 0 else "#d32f2f" if row['Place_Profit'] < 0 else "black"
                                t_color = "#2e7d32" if row['Total P/L'] > 0 else "#d32f2f" if row['Total P/L'] < 0 else "black"
                                
                                if row['Period'] == 'All Time':
                                    sub_df = merged_smart[merged_smart[sys_col_found] == row[sys_col_found]]
                                else:
                                    sub_df = merged_smart[(merged_smart[sys_col_found] == row[sys_col_found]) & (merged_smart['Month_Yr'] == row['Period'])]
                                    
                                csv_b64 = base64.b64encode(sub_df.to_csv(index=False).encode('utf-8')).decode()
                                safe_name = str(row[sys_col_found]).replace(' ', '_').replace('/', '-')
                                safe_per = str(row['Period']).replace(' ', '')
                                dl_link = f'<a href="data:file/csv;base64,{csv_b64}" download="{safe_name}_{safe_per}.csv" style="text-decoration:none; font-size:16px;" title="Download selections">📥</a>'
                                
                                html_table += f"""<tr style="background-color: {bg};"><td class="left-align"><b>{row[sys_col_found]}</b></td><td class="left-align">{b_s}{row['Period']}{b_e}</td><td>{row['Bets']}</td><td>{row['Wins']}</td><td style="color:{w_color}; font-weight:bold;">£{row['Win_Profit']:.2f}</td><td>{row['Strike Rate (%)']:.2f}%</td><td>{row['Places']}</td><td style="color:{p_color}; font-weight:bold;">£{row['Place_Profit']:.2f}</td><td>{row['Place SR (%)']:.2f}%</td><td style="color:{t_color}; font-weight:bold;">£{row['Total P/L']:.2f}</td><td>{dl_link}</td></tr>"""
                            html_table += "</tbody></table></div>"
                            st.markdown(html_table, unsafe_allow_html=True)
                        else: st.warning("Found the file, but none of the picks had a matched race result in the database.")
                    else: st.error("The file is missing one of the required columns: Date, Time, Course, or Horse.")
                except Exception as e: st.error(f"Error processing {target_ods}: {e}")
            else: 
                if st.session_state.get("is_admin") and "Admin" in perf_file_choice:
                    st.info("To see Admin performance tracking, please upload 'BOTManAdminMaster.ods' to the root folder.")
                else:
                    st.info("To see live performance tracking, please upload 'BOTManSystemsMaster.ods' to the root folder.")

# --- Page 4: Mini SYSTEM BUILDER ---
    elif app_mode == "🛠️ System Builder":
        # --- THE DYNAMIC RESET HACK ---
        if "form_reset_counter" not in st.session_state:
            st.session_state.form_reset_counter = 0
        if "sys_defaults" not in st.session_state:
            st.session_state.sys_defaults = {}

        c_title, c_reset = st.columns([4, 1])
        with c_title:
            st.header("🛠️ Mini System Builder")
        with c_reset:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Reset Filters", use_container_width=True):
                # Clear the results table from the screen
                if 'tab4_results' in st.session_state:
                    del st.session_state['tab4_results']
                # Clear any loaded system defaults
                st.session_state.sys_defaults = {}
                # Change the form ID to force a complete visual wipe
                st.session_state.form_reset_counter += 1
                st.rerun()

        # --- THE BENCHMARK TOGGLE (ADMIN ONLY) ---
        if st.session_state.get("is_admin"):
            ai_mode = st.radio("🧠 **AI Backtest Engine (Admin Only):**", 
                              ["💾 Use Prediction Vault (Historical Reality)", "⚡ Use Today's Live Brain (Benchmark Test)"], 
                              horizontal=True, 
                              help="The Vault shows you what the AI predicted on the actual day. The Live Brain applies today's brand new logic to the past.")
            use_vault_bool = "Vault" in ai_mode
        else:
            use_vault_bool = True  # Guests are completely locked into historical reality
            
        st.markdown("---")

        if df_all is not None and not df_all.empty:
            b_df = prep_system_builder_data(df_all, model, feats, shadow_model, shadow_feats, cal_model, is_live_today=False, use_vault=use_vault_bool)

            with st.form(f"builder_form_{st.session_state.form_reset_counter}"):
                st.markdown("### Core Filters")
                
                d_col, m_col = st.columns([1, 3])
                with d_col:
                    if 'Date_DT' in b_df.columns and not b_df['Date_DT'].dropna().empty:
                        min_d = b_df['Date_DT'].min().date()
                        max_d = b_df['Date_DT'].max().date()
                    else:
                        min_d = datetime(2024, 1, 1).date()
                        max_d = datetime.now().date()
                    date_range = st.date_input("Test Specific Period (From - To)", [min_d, max_d], min_value=min_d, max_value=max_d)
                
                # --- PULL DEFAULTS FROM SAVED SYSTEM (IF LOADED) ---
                defs = st.session_state.sys_defaults

                with m_col:
                    all_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                    selected_months = st.multiselect("Include Specific Months (Seasonal Filter)", all_months, default=defs.get('months', all_months))
                
                st.markdown("---")
                course_opts = sorted([str(x).strip() for x in b_df['Course'].dropna().unique() if str(x).strip()])
                safe_courses = [c for c in defs.get('courses', []) if c in course_opts]
                selected_courses = st.multiselect("🎯 Specific Course(s) [Leave blank to include ALL courses]", course_opts, default=safe_courses)
                st.markdown("<br>", unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    race_types = b_df['Race Type'].dropna().unique().tolist()
                    safe_r_types = [r for r in defs.get('race_types', race_types) if r in race_types]
                    selected_race_types = st.multiselect("Race Type", race_types, default=safe_r_types if safe_r_types else race_types)
                    
                    hcap_types = b_df['H/Cap'].dropna().unique().tolist()
                    safe_h_types = [h for h in defs.get('hcap_types', hcap_types) if h in hcap_types]
                    selected_hcap = st.multiselect("Handicap Status", hcap_types, default=safe_h_types if safe_h_types else hcap_types)
                with c2:
                    p_col1, p_col2 = st.columns(2)
                    with p_col1: price_min = st.number_input("Min Price (7:30AM)", 0.0, 1000.0, float(defs.get('price_min', 0.0)), 0.5)
                    with p_col2: price_max = st.number_input("Max Price (7:30AM)", 0.0, 1000.0, float(defs.get('price_max', 1000.0)), 0.5)
                    min_prob_gap = st.number_input("Minimum Prob Gap (%)", -100.0, 50.0, float(defs.get('min_prob_gap', -1.0) * 100), 0.5) / 100
                    min_edge_perc = st.number_input("Min Value Edge % (Leashed)", -100.0, 500.0, float(defs.get('min_edge_perc', -100.0)), step=5.0)
                with c3:
                    rnr_opts = ["2-7", "8-12", "13-16", ">16"]
                    selected_rnrs = st.multiselect("No. of Runners", rnr_opts, default=defs.get('rnrs', rnr_opts))
                    if 'Class' in b_df.columns and not b_df['Class'].dropna().empty:
                        classes = sorted([int(x) for x in b_df['Class'].dropna().unique() if str(x).isdigit() or isinstance(x, (int, float))])
                        safe_classes = [c for c in defs.get('classes', classes) if c in classes]
                        selected_classes = st.multiselect("Class (1-6)", classes, default=safe_classes if safe_classes else classes)
                    else:
                        st.multiselect("Class (1-6)", ["Not Found in CSV"], disabled=True)
                        selected_classes = []
                with c4:
                    if 'Class Move' in b_df.columns and not b_df['Class Move'].dropna().empty:
                        cm_opts = [x for x in b_df['Class Move'].dropna().unique() if x in ['U', 'D', 'S']]
                        safe_cm = [c for c in defs.get('cm', cm_opts) if c in cm_opts]
                        selected_cm = st.multiselect("Class Movement", cm_opts, default=safe_cm if safe_cm else cm_opts)
                    else:
                        st.multiselect("Class Movement", ["Not Found in CSV"], disabled=True)
                        selected_cm = []

                c5, c6, c7, c8 = st.columns(4)
                with c5:
                    rank_1_only = st.checkbox("Must be AI Rank 1", value=defs.get('rank_1_only', False))
                    sex_opts = ["c", "f", "g", "m", "h", "r", "x"]
                    safe_sex = [s for s in defs.get('sex', sex_opts) if s in sex_opts]
                    selected_sex = st.multiselect("Horse Sex", sex_opts, default=safe_sex if safe_sex else sex_opts)
                with c6: 
                    vf_opts = ["Off", "Original AI vs 7:30AM", "Original AI vs BSP", "Calibrated AI (Leashed) vs 7:30AM", "Calibrated AI (Leashed) vs BSP", "My Value vs 7:30AM", "My Value vs BSP"]
                    saved_vf = defs.get('value_filter', "Off")
                    if saved_vf == "Value vs 7:30AM Price": saved_vf = "Original AI vs 7:30AM"
                    if saved_vf == "Value vs BSP": saved_vf = "Original AI vs BSP"
                    if saved_vf == "AI Value vs 7:30AM": saved_vf = "Original AI vs 7:30AM"
                    if saved_vf == "AI Value vs BSP": saved_vf = "Original AI vs BSP"
                    try: vf_idx = vf_opts.index(saved_vf)
                    except ValueError: vf_idx = 0
                    value_filter = st.selectbox("Value Strategy", vf_opts, index=vf_idx)
                with c7: 
                    ir_opts = ["Any", "Y (Yes)", "No (Blank)"]
                    try: ir_idx = ir_opts.index(defs.get('irish', "Any"))
                    except ValueError: ir_idx = 0
                    irish_f = st.selectbox("Irish Race", ir_opts, index=ir_idx)
                with c8: 
                    age_min, age_max = st.slider("Horse Age Range", 1, 20, (int(defs.get('age_min', 1)), int(defs.get('age_max', 20))), 1)
                
                st.markdown("### 📊 Display Options")
                
                master_group_opts = [
                    'Race Type', 'H/Cap', 'Price Bracket', 'Edge Bracket', 'Month_Yr', 'Course', 
                    'Class', 'Class Move', 'No. of Rnrs', 'Age', 'Sex', 'Irish?', 'Irish',
                    'Comb. Rank', 'Comp. Rank', 'Speed Rank', 'Race Rank', 
                    'Primary Rank', 'Form Rank', 'Pure Rank', 'MSAI Rank', 
                    'PRB Rank', 'Trainer PRB Rank', 'Jockey PRB Rank'
                ]
                
                valid_group_opts = [c for c in master_group_opts if c in b_df.columns or c == 'Month_Yr']
                saved_groupings = defs.get('groupby', ['Race Type', 'H/Cap', 'Price Bracket'])
                safe_groupings = [g for g in saved_groupings if g in valid_group_opts]
                
                if not safe_groupings and 'Race Type' in valid_group_opts:
                    safe_groupings = ['Race Type']
                    
                selected_groupby = st.multiselect("Group Breakdown Table By (Select up to 3):", valid_group_opts, default=safe_groupings, max_selections=3)

                with st.expander("📊 Advanced Rank Filters", expanded=False):
                    rank_opts = ["Any", "Rank 1", "Top 2", "Top 3", "Top 4", "Top 5"]
                    def get_r_idx(col_name):
                        val = defs.get('ranks', {}).get(col_name, "Any")
                        return rank_opts.index(val) if val in rank_opts else 0
                        
                    r1_c1, r1_c2, r1_c3, r1_c4, r1_c5 = st.columns(5)
                    with r1_c1: comb_f = st.selectbox("Comb. Rank", rank_opts, index=get_r_idx('Comb. Rank'))
                    with r1_c2: comp_f = st.selectbox("Comp. Rank", rank_opts, index=get_r_idx('Comp. Rank'))
                    with r1_c3: speed_f = st.selectbox("Speed Rank", rank_opts, index=get_r_idx('Speed Rank'))
                    with r1_c4: race_f = st.selectbox("Race Rank", rank_opts, index=get_r_idx('Race Rank'))
                    with r1_c5: primary_f = st.selectbox("Primary Rank", rank_opts, index=get_r_idx('Primary Rank'))
                    
                    r2_c1, r2_c2, r2_c3, r2_c4, r2_c5 = st.columns(5)
                    with r2_c1: msai_f = st.selectbox("MSAI Rank", rank_opts, index=get_r_idx('MSAI Rank'))
                    with r2_c2: prb_f = st.selectbox("PRB Rank", rank_opts, index=get_r_idx('PRB Rank'))
                    with r2_c3: tprb_f = st.selectbox("Trainer PRB Rank", rank_opts, index=get_r_idx('Trainer PRB Rank'))
                    with r2_c4: jprb_f = st.selectbox("Jockey PRB Rank", rank_opts, index=get_r_idx('Jockey PRB Rank'))
                    with r2_c5: form_f = st.selectbox("Form Rank", rank_opts, index=get_r_idx('Form Rank'))
                    
                    r3_c1, r3_c2, r3_c3, r3_c4, r3_c5 = st.columns(5)
                    with r3_c1: pure_f = st.selectbox("Pure Rank", rank_opts, index=get_r_idx('Pure Rank'))
                
                submit_button = st.form_submit_button(label="🚀 Process Data")

            if st.session_state.get("is_admin"):
                st.markdown("---")
                st.markdown("### ⚙️ Admin: Generate System Code (For GitHub)")
                c_name, c_btn = st.columns([3, 1])
                with c_name: new_sys_name = st.text_input("System Name:", placeholder="e.g., A/W MSAI Value")
                with c_btn:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Generate JSON Code", use_container_width=True):
                        if new_sys_name:
                            sys_data = {
                                "race_types": selected_race_types, "hcap_types": selected_hcap, "price_min": price_min, "price_max": price_max, "min_prob_gap": min_prob_gap, "min_edge_perc": min_edge_perc, "rnrs": selected_rnrs, "classes": selected_classes, "cm": selected_cm, "sex": selected_sex, "courses": selected_courses, "rank_1_only": rank_1_only, "value_filter": value_filter, "irish": irish_f, "age_min": age_min, "age_max": age_max, "months": selected_months,
                                "ranks": {"Comb. Rank": comb_f, "Comp. Rank": comp_f, "Speed Rank": speed_f, "Race Rank": race_f, "Primary Rank": primary_f, "MSAI Rank": msai_f, "PRB Rank": prb_f, "Trainer PRB Rank": tprb_f, "Jockey PRB Rank": jprb_f, "Form Rank": form_f, "Pure Rank": pure_f},
                                "groupby": selected_groupby
                            }
                            st.code(f'"{new_sys_name}": {json.dumps(sys_data, indent=4)}', language="json")
                        else: st.error("Please enter a name for the system to generate code.")
                
                c_pub, c_sec = st.columns(2)
                with c_pub:
                    if os.path.exists("BOTMan_user_systems.json"):
                        with st.expander("📖 View Active PUBLIC Systems"):
                            try:
                                with open("BOTMan_user_systems.json", "r") as f: saved_dict = json.load(f)
                                if saved_dict:
                                    for s_key, s_data in list(saved_dict.items()):
                                        with st.expander(f"🔍 {s_key}"): 
                                            st.json(s_data)
                                            if st.button(f"📥 Load '{s_key}' into Builder", key=f"load_pub_{s_key}", use_container_width=True):
                                                st.session_state.sys_defaults = s_data
                                                st.session_state.form_reset_counter += 1
                                                if 'tab4_results' in st.session_state: del st.session_state['tab4_results']
                                                st.rerun()
                                else: st.write("No systems currently active.")
                            except Exception as e: st.error(f"Error reading public file: {e}")
                
                with c_sec:
                    if os.path.exists("BOTMan_admin_systems.json"):
                        with st.expander("🕵️ View Active ADMIN SECRETS"):
                            try:
                                with open("BOTMan_admin_systems.json", "r") as f: admin_dict = json.load(f)
                                if admin_dict:
                                    for s_key, s_data in list(admin_dict.items()):
                                        with st.expander(f"🔍 {s_key}"): 
                                            st.json(s_data)
                                            if st.button(f"📥 Load '{s_key}' into Builder", key=f"load_sec_{s_key}", use_container_width=True):
                                                st.session_state.sys_defaults = s_data
                                                st.session_state.form_reset_counter += 1
                                                if 'tab4_results' in st.session_state: del st.session_state['tab4_results']
                                                st.rerun()
                                else: st.write("No admin systems currently active.")
                            except Exception as e: st.error(f"Error reading admin file: {e}")
                st.markdown("---")

            if submit_button:
                st.success("✅ System recalculated instantly!")

                mask = (b_df['Race Type'].isin(selected_race_types) & b_df['H/Cap'].isin(selected_hcap) & (b_df['7:30AM Price'] >= price_min) & (b_df['7:30AM Price'] <= price_max) & (b_df['Prob Gap'] >= min_prob_gap) & (b_df['Value_Edge_Perc'] >= min_edge_perc))
                
                if len(date_range) == 2: mask = mask & (b_df['Date_DT'].dt.date >= date_range[0]) & (b_df['Date_DT'].dt.date <= date_range[1])
                elif len(date_range) == 1: mask = mask & (b_df['Date_DT'].dt.date == date_range[0])
                
                if len(selected_months) < 12:
                    month_map = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
                    sel_m_nums = [month_map[m] for m in selected_months]
                    mask = mask & b_df['Date_DT'].dt.month.isin(sel_m_nums)
                
                if rank_1_only: mask = mask & (b_df['Rank'] == 1)
                
                if "Original AI" in value_filter: 
                    v_col = 'Value Price'
                    m_col = 'BSP' if "BSP" in value_filter else '7:30AM Price'
                    mask = mask & (b_df[m_col] > b_df[v_col])
                elif "Calibrated AI" in value_filter:
                    v_col = 'Cal_Value_Price'
                    m_col = 'BSP' if "BSP" in value_filter else '7:30AM Price'
                    mask = mask & (b_df[m_col] > b_df[v_col])
                elif "My Value" in value_filter:
                    m_col = 'BSP' if "BSP" in value_filter else '7:30AM Price'
                    mask = mask & (b_df[m_col] > b_df['User Value'])
                
                rnr_mask = pd.Series(False, index=b_df.index)
                if "2-7" in selected_rnrs: rnr_mask |= (b_df['No. of Rnrs'] >= 2) & (b_df['No. of Rnrs'] <= 7)
                if "8-12" in selected_rnrs: rnr_mask |= (b_df['No. of Rnrs'] >= 8) & (b_df['No. of Rnrs'] <= 12)
                if "13-16" in selected_rnrs: rnr_mask |= (b_df['No. of Rnrs'] >= 13) & (b_df['No. of Rnrs'] <= 16)
                if ">16" in selected_rnrs: rnr_mask |= (b_df['No. of Rnrs'] > 16)
                mask = mask & rnr_mask

                if 'Class' in b_df.columns and selected_classes: mask = mask & b_df['Class'].isin(selected_classes)
                if 'Class Move' in b_df.columns and selected_cm: mask = mask & b_df['Class Move'].isin(selected_cm)
                if 'Age' in b_df.columns: mask = mask & (b_df['Age'] >= age_min) & (b_df['Age'] <= age_max)
                if 'Sex' in b_df.columns and selected_sex: mask = mask & b_df['Sex'].astype(str).str.strip().str.lower().isin([s.lower() for s in selected_sex])
                if 'Course' in b_df.columns and selected_courses: mask = mask & b_df['Course'].astype(str).str.strip().isin(selected_courses)

                t_irish_col = 'Irish?' if 'Irish?' in b_df.columns else 'Irish' if 'Irish' in b_df.columns else None
                if t_irish_col and irish_f != "Any":
                    t_irish_series = b_df[t_irish_col].astype(str).str.strip().str.upper()
                    if irish_f == "Y (Yes)": mask = mask & (t_irish_series == 'Y')
                    elif irish_f == "No (Blank)": mask = mask & (t_irish_series != 'Y')

                def apply_rank_filter(df_mask, current_df, col_name, setting):
                    if setting != "Any" and col_name in current_df.columns:
                        num_col = pd.to_numeric(current_df[col_name], errors='coerce')
                        if setting == "Rank 1": return df_mask & (num_col == 1)
                        elif setting == "Top 2": return df_mask & (num_col <= 2)
                        elif setting == "Top 3": return df_mask & (num_col <= 3)
                        elif setting == "Top 4": return df_mask & (num_col <= 4)
                        elif setting == "Top 5": return df_mask & (num_col <= 5)
                    return df_mask

                mask = apply_rank_filter(mask, b_df, 'Comb. Rank', comb_f)
                mask = apply_rank_filter(mask, b_df, 'Comp. Rank', comp_f)
                mask = apply_rank_filter(mask, b_df, 'Speed Rank', speed_f)
                mask = apply_rank_filter(mask, b_df, 'Race Rank', race_f)
                mask = apply_rank_filter(mask, b_df, 'Primary Rank', primary_f)
                mask = apply_rank_filter(mask, b_df, 'MSAI Rank', msai_f)
                mask = apply_rank_filter(mask, b_df, 'PRB Rank', prb_f)
                mask = apply_rank_filter(mask, b_df, 'Trainer PRB Rank', tprb_f)
                mask = apply_rank_filter(mask, b_df, 'Jockey PRB Rank', jprb_f)
                mask = apply_rank_filter(mask, b_df, 'Form Rank', form_f)
                mask = apply_rank_filter(mask, b_df, 'Pure Rank', pure_f)
                
                df_filtered = b_df[mask].copy()

                if not df_filtered.empty:
                    hist_csv_data_out = df_filtered.to_csv(index=False).encode('utf-8')
                    sys_timestamp = datetime.now().strftime('%d%m%y_%H%M%S')

                    if 'Month_Yr' not in df_filtered.columns and 'Date_DT' in df_filtered.columns:
                        df_filtered['Month_Yr'] = df_filtered['Date_DT'].dt.strftime('%Y-%m')

                    if not selected_groupby: selected_groupby = ['Race Type', 'H/Cap', 'Price Bracket']

                    breakdown = df_filtered.groupby(selected_groupby, observed=False).agg(
                        Bets=('Horse', 'count'), Wins=('Is_Win', 'sum'), Win_Profit=('Win P/L <2%', 'sum'), Places=('Is_Place', 'sum'), Place_Profit=('Place P/L <2%', 'sum')
                    ).reset_index()
                    
                    breakdown = breakdown[breakdown['Bets'] > 0]
                    breakdown['Strike Rate (%)'] = (breakdown['Wins'] / breakdown['Bets'] * 100).fillna(0)
                    breakdown['Place SR (%)'] = (breakdown['Places'] / breakdown['Bets'] * 100).fillna(0)
                    breakdown['Win ROI (%)'] = (breakdown['Win_Profit'] / breakdown['Bets'] * 100).fillna(0)
                    breakdown['Total P/L'] = breakdown['Win_Profit'] + breakdown['Place_Profit']
                    breakdown = breakdown.sort_values(by=selected_groupby)

                    # Calculate Base KPIs
                    total_sys_bets = breakdown['Bets'].sum()
                    total_sys_profit = breakdown['Total P/L'].sum()
                    total_wins = breakdown['Wins'].sum()
                    total_places = breakdown['Places'].sum()
                    total_win_profit = breakdown['Win_Profit'].sum()
                    total_place_profit = breakdown['Place_Profit'].sum()
                    overall_roi = (total_sys_profit / total_sys_bets * 100) if total_sys_bets > 0 else 0
                    
                    kpis = [
                        total_sys_bets, total_wins, total_places, 
                        total_win_profit, total_place_profit, overall_roi
                    ]
                    
                    # --- ADVANCED KPIs (LLR & MAX DD) ---
                    chron_df = df_filtered.sort_values(by=['Date_DT', 'Time'])
                    
                    # Longest Losing Run (LLR) based on Win ONLY
                    is_loss = (chron_df['Is_Win'] == 0).astype(int)
                    losing_streaks = is_loss.groupby((is_loss != is_loss.shift()).cumsum()).sum()
                    llr = int(losing_streaks.max()) if not losing_streaks.empty else 0
                    
                    # Maximum Drawdown (Max DD) based on Win ONLY
                    cum_profit = chron_df['Win P/L <2%'].cumsum()
                    running_max = cum_profit.cummax()
                    drawdowns = running_max - cum_profit
                    max_dd = float(drawdowns.max()) if not drawdowns.empty else 0.0
                    
                    # Strike Rates & ROI
                    win_sr = (total_wins / total_sys_bets * 100) if total_sys_bets > 0 else 0.0
                    win_roi = (total_win_profit / total_sys_bets * 100) if total_sys_bets > 0 else 0.0
                    place_sr = (total_places / total_sys_bets * 100) if total_sys_bets > 0 else 0.0
                    place_roi = (total_place_profit / total_sys_bets * 100) if total_sys_bets > 0 else 0.0
                    
                    adv_kpis = [win_sr, win_roi, place_sr, place_roi, llr, max_dd]

                    # --- NEW: ELITE QUANT METRICS ---
                    # 1. Expected Wins (Sum of implied probabilities from 7:30AM Prices)
                    valid_prices = chron_df['7:30AM Price'][chron_df['7:30AM Price'] > 1.0]
                    exp_wins = (1 / valid_prices).sum() if not valid_prices.empty else 0.0
                    
                    # 2. A/E Ratio (Actual / Expected)
                    ae_ratio = (total_wins / exp_wins) if exp_wins > 0 else 0.0
                    
                    # 3. Chi Score (Full Statistical Formula)
                    if exp_wins > 0 and total_sys_bets > exp_wins:
                        exp_losses = total_sys_bets - exp_wins
                        actual_losses = total_sys_bets - total_wins
                        win_chi = ((total_wins - exp_wins)**2) / exp_wins
                        loss_chi = ((actual_losses - exp_losses)**2) / exp_losses
                        chi_score = win_chi + loss_chi
                    else:
                        chi_score = 0.0
                    
                    # 4. Sortino Ratio (Industry Standard: Annualized Daily Returns)
                    daily_returns = chron_df.groupby(chron_df['Date_DT'].dt.date)['Win P/L <2%'].sum()
                    mean_daily = daily_returns.mean() if not daily_returns.empty else 0.0
                    downside_daily = daily_returns[daily_returns < 0]
                    downside_var = (downside_daily**2).mean()
                    downside_std = downside_var**0.5 if pd.notna(downside_var) and downside_var > 0 else 0.0001
                    
                    # Scale by sqrt(365) to annualize the daily racing returns
                    sortino = (mean_daily / downside_std) * (365**0.5) if not daily_returns.empty else 0.0
                    
                    # 5. Ulcer Index (Root Mean Square of Drawdowns)
                    ulcer_var = (drawdowns**2).mean()
                    ulcer = ulcer_var**0.5 if pd.notna(ulcer_var) else 0.0
                    
                    quant_kpis = [ae_ratio, chi_score, sortino, ulcer]

                    qual_html_out, csv_data_out, timestamp_out = "", None, ""
                    val_bsp_warning = value_filter in ["Original AI vs BSP", "Calibrated AI (Leashed) vs BSP", "My Value vs BSP"]

                    if df_today is not None and not df_today.empty:
                        t_df = prep_system_builder_data(df_today, model, feats, shadow_model, shadow_feats, cal_model, is_live_today=True)
                        
                        t_mask = (t_df['Race Type'].isin(selected_race_types) & t_df['H/Cap'].isin(selected_hcap) & (t_df['7:30AM Price'] >= price_min) & (t_df['7:30AM Price'] <= price_max) & (t_df['Prob Gap'] >= min_prob_gap) & (t_df['Value_Edge_Perc'] >= min_edge_perc))
                        
                        if len(selected_months) < 12:
                            t_mask = t_mask & t_df['Date_DT'].dt.month.isin(sel_m_nums)

                        if rank_1_only: t_mask = t_mask & (t_df['Rank'] == 1)
                        
                        if "Original AI" in value_filter: 
                            v_col = 'Value Price'
                            m_col = 'BSP' if "BSP" in value_filter else '7:30AM Price'
                            t_mask = t_mask & (t_df[m_col] > t_df[v_col])
                        elif "Calibrated AI" in value_filter:
                            v_col = 'Cal_Value_Price'
                            m_col = 'BSP' if "BSP" in value_filter else '7:30AM Price'
                            t_mask = t_mask & (t_df[m_col] > t_df[v_col])
                        elif "My Value" in value_filter:
                            m_col = 'BSP' if "BSP" in value_filter else '7:30AM Price'
                            t_mask = t_mask & (t_df[m_col] > t_df['User Value'])
                        
                        t_rnr_mask = pd.Series(False, index=t_df.index)
                        if "2-7" in selected_rnrs: t_rnr_mask |= (t_df['No. of Rnrs'] >= 2) & (t_df['No. of Rnrs'] <= 7)
                        if "8-12" in selected_rnrs: t_rnr_mask |= (t_df['No. of Rnrs'] >= 8) & (t_df['No. of Rnrs'] <= 12)
                        if "13-16" in selected_rnrs: t_rnr_mask |= (t_df['No. of Rnrs'] >= 13) & (t_df['No. of Rnrs'] <= 16)
                        if ">16" in selected_rnrs: t_rnr_mask |= (t_df['No. of Rnrs'] > 16)
                        t_mask = t_mask & t_rnr_mask

                        if 'Class' in t_df.columns and selected_classes: t_mask = t_mask & t_df['Class'].isin(selected_classes)
                        if 'Class Move' in t_df.columns and selected_cm: t_mask = t_mask & t_df['Class Move'].isin(selected_cm)
                        if 'Age' in t_df.columns: t_mask = t_mask & (t_df['Age'] >= age_min) & (t_df['Age'] <= age_max)
                        if 'Sex' in t_df.columns and selected_sex: t_mask = t_mask & t_df['Sex'].astype(str).str.strip().str.lower().isin([s.lower() for s in selected_sex])
                        if 'Course' in t_df.columns and selected_courses: t_mask = t_mask & t_df['Course'].astype(str).str.strip().isin(selected_courses)

                        if t_irish_col and irish_f != "Any":
                            t_irish_series = t_df[t_irish_col].astype(str).str.strip().str.upper()
                            if irish_f == "Y (Yes)": t_mask = t_mask & (t_irish_series == 'Y')
                            elif irish_f == "No (Blank)": t_mask = t_mask & (t_irish_series != 'Y')

                        t_mask = apply_rank_filter(t_mask, t_df, 'Comb. Rank', comb_f)
                        t_mask = apply_rank_filter(t_mask, t_df, 'Comp. Rank', comp_f)
                        t_mask = apply_rank_filter(t_mask, t_df, 'Speed Rank', speed_f)
                        t_mask = apply_rank_filter(t_mask, t_df, 'Race Rank', race_f)
                        t_mask = apply_rank_filter(t_mask, t_df, 'Primary Rank', primary_f)
                        t_mask = apply_rank_filter(t_mask, t_df, 'MSAI Rank', msai_f)
                        t_mask = apply_rank_filter(t_mask, t_df, 'PRB Rank', prb_f)
                        t_mask = apply_rank_filter(t_mask, t_df, 'Trainer PRB Rank', tprb_f)
                        t_mask = apply_rank_filter(t_mask, t_df, 'Jockey PRB Rank', jprb_f)
                        t_mask = apply_rank_filter(t_mask, t_df, 'Form Rank', form_f)
                        t_mask = apply_rank_filter(t_mask, t_df, 'Pure Rank', pure_f)
                        
                        t_filtered = t_df[t_mask].copy()
                        
                        if not t_filtered.empty:
                            t_filtered = t_filtered.sort_values(by=['Time', 'Course'])
                            dl_cols = [c for c in ['Date', 'Time', 'Course', 'Horse', '7:30AM Price', 'ML_Prob', 'Rank', 'Pure Rank'] if c in t_filtered.columns]
                            csv_data_out = t_filtered[dl_cols].to_csv(index=False).encode('utf-8')
                            timestamp_out = datetime.now().strftime('%d%m%y_%H%M%S')
                            
                            qual_html_out = '<div class="scrollable-table"><table class="builder-table"><thead><tr style="background-color: #2e7d32; color: white;"><th class="center-text">Date</th><th class="center-text">Time</th><th class="left-align">Course</th><th class="left-align">Horse</th><th class="center-text">7:30AM Price</th><th class="center-text">Pure Rank</th></tr></thead><tbody>'
                            for _, q_row in t_filtered.iterrows(): qual_html_out += f"<tr><td class='center-text'>{q_row['Date']}</td><td class='center-text'>{q_row['Time']}</td><td class='left-align'>{q_row['Course']}</td><td class='left-align'><b>{q_row['Horse']}</b></td><td class='center-text'>{q_row['7:30AM Price']:.2f}</td><td class='center-text'><b>{int(q_row.get('Pure Rank', 0))}</b></td></tr>"
                            qual_html_out += "</tbody></table></div>"

                    html_table_out = '<style>.builder-table { border-collapse: collapse; width: 100%; min-width: 900px; font-size: 14px; font-family: sans-serif; } .builder-table th, .builder-table td { border: 1px solid #ccc; padding: 4px; text-align: center; white-space: nowrap; } .builder-table tr:hover { background-color: #0000FF !important; color: white !important; } .left-align { text-align: left !important; padding-left: 8px !important; }</style>'
                    html_table_out += '<div class="scrollable-table"><table class="builder-table"><thead><tr style="background-color: #f0f2f6; color: black;">'
                    
                    for col in selected_groupby:
                        html_table_out += f'<th class="left-align">{col}</th>'
                        
                    html_table_out += '<th>Bets</th><th>Wins</th><th>Win P/L</th><th>Win SR</th><th>Places</th><th>Plc P/L</th><th>Plc SR</th><th>Total P/L</th></tr></thead><tbody>'
                    
                    for _, row in breakdown.iterrows(): 
                        t_col = "#2e7d32" if row['Total P/L'] >= 0 else "#d32f2f"
                        html_table_out += "<tr>"
                        
                        for col in selected_groupby:
                            val = row[col]
                            if isinstance(val, float) and val.is_integer(): val = int(val)
                            html_table_out += f"<td class='left-align'>{val}</td>"
                            
                        html_table_out += f"<td>{row['Bets']}</td><td>{row['Wins']}</td><td><b>£{row['Win_Profit']:.2f}</b></td><td>{row['Strike Rate (%)']:.2f}%</td><td>{row['Places']}</td><td><b>£{row['Place_Profit']:.2f}</b></td><td>{row['Place SR (%)']:.2f}%</td><td style='color:{t_col};'><b>£{row['Total P/L']:.2f}</b></td></tr>"
                    
                    html_table_out += "</tbody></table></div>"

                    st.session_state['tab4_results'] = {
                        'kpis': kpis, 'adv_kpis': adv_kpis, 'quant_kpis': quant_kpis, 'breakdown_html': html_table_out, 'qual_html': qual_html_out, 
                        'csv': csv_data_out, 'timestamp': timestamp_out if timestamp_out else sys_timestamp, 
                        'val_warn': val_bsp_warning, 'hist_csv': hist_csv_data_out
                    }
                else: st.session_state['tab4_results'] = "empty"

            if 'tab4_results' in st.session_state:
                if st.session_state['tab4_results'] == "empty": st.warning("No bets found matching these exact criteria.")
                else:
                    res = st.session_state['tab4_results']
                    kpis = res['kpis']
                    adv_kpis = res.get('adv_kpis', [0.0, 0.0, 0.0, 0.0, 0, 0.0])
                    quant = res.get('quant_kpis', [0.0, 0.0, 0.0, 0.0])
                    
                    st.markdown("### System Preview Performance")
                    
                    # Row 1: Base KPIs
                    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
                    kpi1.metric("Total Bets", kpis[0])
                    kpi2.metric("Wins", kpis[1])
                    kpi3.metric("Places", kpis[2])
                    kpi4.metric("Win P/L", f"£{kpis[3]:.2f}")
                    kpi5.metric("Place P/L", f"£{kpis[4]:.2f}")
                    kpi6.metric("Total ROI", f"{kpis[5]:.2f}%")
                    
                    # Row 2: Advanced KPIs
                    st.markdown("<br>", unsafe_allow_html=True)
                    r2_1, r2_2, r2_3, r2_4, r2_5, r2_6 = st.columns(6)
                    r2_1.metric("Win S/R", f"{adv_kpis[0]:.1f}%")
                    r2_2.metric("Win ROI", f"{adv_kpis[1]:.1f}%")
                    r2_3.metric("Place S/R", f"{adv_kpis[2]:.1f}%")
                    r2_4.metric("Place ROI", f"{adv_kpis[3]:.1f}%")
                    r2_5.metric("LLR", int(adv_kpis[4]), help="Longest Losing Run (Consecutive losers)")
                    r2_6.metric("Max. DD", f"£{adv_kpis[5]:.2f}", help="Maximum Drawdown (Biggest drop from peak profit)")

                    # Row 3: Quant Metrics
                    st.markdown("<br>", unsafe_allow_html=True)
                    q1, q2, q3, q4 = st.columns(4)
                    q1.metric("A/E Ratio", f"{quant[0]:.2f}", help="Actual vs Expected Wins. > 1.0 means the system beats the market. > 1.20 is elite.")
                    q2.metric("Chi Score", f"{quant[1]:.1f}", help="Checks if profit is just luck. > 3.84 means 95% confidence it's real. > 6.63 is 99% confidence.")
                    q3.metric("Sortino Ratio", f"{quant[2]:.2f}", help="Measures profit vs. 'bad' risk. > 1.0 is good. > 2.0 means the system is an incredibly smooth profit generator.")
                    q4.metric("Ulcer Index", f"{quant[3]:.1f}", help="Average drawdown depth in £/pts. The Scale: [0-10: Elite/Low Stress] | [10-25: Normal Variance] | [25-50: High Stress] | [50+: Unplayable Rollercoaster].")

                    if res['qual_html'] != "":
                        st.markdown("<br>", unsafe_allow_html=True)
                        with st.expander("🔍 View Today's Qualifiers for this System", expanded=False):
                            if res['val_warn']: st.info("ℹ️ **Note:** You selected a 'vs BSP' value filter. Because today's BSP is not yet known, the live qualifiers are falling back to use the 7:30AM Price to check for Value.")
                            st.download_button(label="📥 Download Qualifiers to CSV", data=res['csv'], file_name=f"BOTMan_System_Preview_{res['timestamp']}.csv", mime="text/csv")
                            st.write("") 
                            st.markdown(res['qual_html'], unsafe_allow_html=True)
                    elif df_today is not None and not df_today.empty:
                        st.markdown("<br>", unsafe_allow_html=True)
                        with st.expander("🔍 View Today's Qualifiers for this System", expanded=False):
                            st.info("There are no horses running today that match these exact system criteria.")

                    st.markdown("### Detailed Preview Breakdown")
                    c_dl, c_blank = st.columns([1, 3])
                    with c_dl:
                        if 'hist_csv' in res and res['hist_csv']:
                            st.download_button(
                                label="📥 Download ALL Historic Data (CSV)",
                                data=res['hist_csv'],
                                file_name=f"BOTMan_Historic_System_{res['timestamp']}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                    st.markdown(res['breakdown_html'], unsafe_allow_html=True)
    
# --- Page 5: RACE ANALYSIS ---
    elif app_mode == "🏇 Race Analysis":
        c_head, c_dl = st.columns([3, 1])
        with c_head:
            st.header("🏇 Race Analysis")
        
        st.markdown('''<style>
            div[data-testid="stButton"] button p {
                white-space: pre-wrap !important;
                text-align: center !important;
                line-height: 1.5 !important;
            }
        </style>''', unsafe_allow_html=True)
        
        if df_today is not None and not df_today.empty:
            ta_df = df_today.copy()
            
            with c_dl:
                st.markdown("<br>", unsafe_allow_html=True)
                dl_cols = ['Date', 'Time', 'Course', 'Horse', 'Primary Rank', 'Form Rank', 'Pure Rank']
                avail_cols = [c for c in dl_cols if c in ta_df.columns]
                
                dl_df = ta_df[avail_cols].copy()
                for col in ['Primary Rank', 'Form Rank', 'Pure Rank']:
                    if col in dl_df.columns:
                        dl_df[col] = pd.to_numeric(dl_df[col], errors='coerce').fillna(0).astype(int)
                
                csv_data = dl_df.to_csv(index=False).encode('utf-8')
                ts = datetime.now().strftime('%d%m%y_%H%M%S')
                st.download_button(
                    label="📥 Download Master CSV",
                    data=csv_data,
                    file_name=f"BOTMan_Race_Analysis_{ts}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            ta_df['Time'] = ta_df['Time'].astype(str).str.strip()
            ta_df['Course'] = ta_df['Course'].astype(str).str.strip()
            
            if st.session_state.get('analysis_race'):
                sel_c = st.session_state.analysis_race['course']
                sel_t = st.session_state.analysis_race['time']
                
                race_info = ta_df[(ta_df['Course'] == sel_c) & (ta_df['Time'] == sel_t)]
                r_type_str = str(race_info['Race Type'].iloc[0]).strip() if not race_info.empty else "Unknown"
                r_hcap_str = "Hcap" if not race_info.empty and str(race_info['H/Cap'].iloc[0]).strip() == 'Y' else "Non-Hcap"
                
                st.markdown(f"### DETAILED RACE ANALYSIS: {sel_c} | {sel_t} | {r_type_str} ({r_hcap_str})")
                
                all_races_df = ta_df[['Time', 'Course']].drop_duplicates().sort_values(['Time', 'Course'])
                all_races = list(zip(all_races_df['Time'], all_races_df['Course']))
                curr_r_idx = all_races.index((sel_t, sel_c)) if (sel_t, sel_c) in all_races else -1
                
                prev_r = all_races[curr_r_idx - 1] if curr_r_idx > 0 else None
                next_r = all_races[curr_r_idx + 1] if curr_r_idx != -1 and curr_r_idx < len(all_races) - 1 else None
                
                meeting_races_df = ta_df[ta_df['Course'] == sel_c][['Time']].drop_duplicates().sort_values('Time')
                meeting_races = meeting_races_df['Time'].tolist()
                curr_m_idx = meeting_races.index(sel_t) if sel_t in meeting_races else -1
                
                prev_m = meeting_races[curr_m_idx - 1] if curr_m_idx > 0 else None
                next_m = meeting_races[curr_m_idx + 1] if curr_m_idx != -1 and curr_m_idx < len(meeting_races) - 1 else None

                nav_cols = st.columns(5)
                with nav_cols[0]:
                    if prev_r:
                        if st.button(f"⏪ <R ({prev_r[0]})", use_container_width=True):
                            st.session_state.analysis_race = {'course': prev_r[1], 'time': prev_r[0]}
                            st.rerun()
                with nav_cols[1]:
                    if prev_m:
                        if st.button(f"◀ <M ({prev_m})", type="primary", use_container_width=True):
                            st.session_state.analysis_race = {'course': sel_c, 'time': prev_m}
                            st.rerun()
                with nav_cols[2]:
                    if st.button("🔙 Back to Race Selection", use_container_width=True):
                        st.session_state.analysis_race = None
                        st.rerun()
                with nav_cols[3]:
                    if next_m:
                        if st.button(f"M> ({next_m}) ▶", type="primary", use_container_width=True):
                            st.session_state.analysis_race = {'course': sel_c, 'time': next_m}
                            st.rerun()
                with nav_cols[4]:
                    if next_r:
                        if st.button(f"R> ({next_r[0]}) ⏩", use_container_width=True):
                            st.session_state.analysis_race = {'course': next_r[1], 'time': next_r[0]}
                            st.rerun()
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                show_msai = False
                if r_type_str in ['A/W', 'Turf']:
                    irish_col = 'Irish?' if 'Irish?' in race_info.columns else 'Irish' if 'Irish' in race_info.columns else None
                    is_irish = (race_info[irish_col].astype(str).str.strip().str.upper() == 'Y').any() if irish_col else False
                    
                    if not is_irish and 'MSAI Rank' in race_info.columns:
                        if pd.to_numeric(race_info['MSAI Rank'], errors='coerce').fillna(0).max() > 0:
                            show_msai = True
                
                st.markdown("##### ↕️ Sort Race Data")
                
                if "saved_sort_by" not in st.session_state: st.session_state.saved_sort_by = "Pure Rank"
                if "saved_sort_dir" not in st.session_state: st.session_state.saved_sort_dir = "Ascending 🔼"
                
                def save_sort_prefs():
                    st.session_state.saved_sort_by = st.session_state.temp_sort_by
                    st.session_state.saved_sort_dir = st.session_state.temp_sort_dir

                sort_c1, sort_c2 = st.columns([2, 1])
                
                sort_options = ["Pure Rank", "No. of Top", "Primary Rank", "Form Rank", "Comb. Rank", "Speed Rank", "Race Rank", "Comp. Rank", "PRB Rank", "Race Rating", "7:30AM Price", "Value Price", "Total", "Speed", "Ability", "Going", "Distance", "Course/Sim", "TrainrF", "JockyF", "Draw"]
                if show_msai: sort_options.append("MSAI Rank")
                avail_sorts = [c for c in sort_options if c in race_info.columns]
                
                with sort_c1:
                    try: default_idx = avail_sorts.index(st.session_state.saved_sort_by)
                    except ValueError: default_idx = 0
                    sort_by = st.selectbox("Sort Table By:", avail_sorts, index=default_idx, key="temp_sort_by", on_change=save_sort_prefs)
                with sort_c2:
                    dir_options = ["Ascending 🔼", "Descending 🔽"]
                    try: dir_idx = dir_options.index(st.session_state.saved_sort_dir)
                    except ValueError: dir_idx = 0
                    sort_dir = st.radio("Order:", dir_options, index=dir_idx, horizontal=True, key="temp_sort_dir", on_change=save_sort_prefs)
                
                is_asc = True if "Ascending" in sort_dir else False
                
                race_df = race_info.copy()
                
                if sort_by in race_df.columns:
                    race_df[sort_by] = pd.to_numeric(race_df[sort_by], errors='coerce').fillna(999 if is_asc else -999)
                
                race_df = race_df.sort_values(by=[sort_by, 'Rank'], ascending=[is_asc, True])
                
                def gv(r, c, num=False, default="-"):
                    v = r.get(c, default)
                    if pd.isna(v) or v == "": return default
                    if num:
                        try: return float(v)
                        except: return default
                    return v
                
                def rc(v):
                    try:
                        v = int(float(v))
                        if v == 1: return "r1"
                        if v == 2: return "r2"
                        if v == 3: return "r3"
                    except: pass
                    return ""

                def fmt_int(v):
                    try: return str(int(float(v)))
                    except: return "-"

                def fmt_2dp(v):
                    try: return f"{float(v):.2f}"
                    except: return "-"

                html = '<div class="scrollable-table"><table class="k2-table" style="width:100%; min-width:1400px;"><thead><tr style="background-color: #1a3a5f; color: white;">'
                
                headers = ["Horse", "Value", "7:30am<br>Price", "Speed<br>Rank", "Comb.<br>Rank", "Race<br>Rank", "Race<br>Rating", "Comp.<br>Rank", "PRB<br>Rank"]
                if show_msai: headers.append("MSAI<br>Rank")
                
                headers.extend(["No. of<br>Top", "Primary<br>Rank", "Form<br>Rank"])
                
                if 'No. of Top' in race_df.columns:
                    race_df['No. of Top'] = pd.to_numeric(race_df['No. of Top'], errors='coerce').fillna(0).astype(int)
                
                for h in headers: 
                    col_style = ' style="width: 12%;"' if h == "Horse" else ''
                    align_class = "left-head" if h == "Horse" else "center-text"
                    html += f'<th rowspan="2" class="{align_class}"{col_style}>{h}</th>'
                
                html += '<th colspan="9" class="center-text" style="border-bottom: 1px dashed #ccc; letter-spacing: 2px; color: #a9bacd;">----------------------- FORM -----------------------</th>'
                html += '<th rowspan="2" class="center-text" style="background-color: #000;">Pure<br>Rank</th></tr><tr style="background-color: #1a3a5f; color: white;">'
                
                for h in ["Ability", "Going", "Distance", "Course/<br>Sim", "Trainer", "Jockey", "Draw", "Speed", "Total"]: 
                    html += f'<th class="center-text">{h}</th>'
                html += '</tr></thead><tbody>'
                
                for _, r in race_df.iterrows():
                    vp, pr = gv(r,"Value Price",True), gv(r,"7:30AM Price",True)
                    sr, cr, rr, cpr, prb = gv(r,"Speed Rank"), gv(r,"Comb. Rank"), gv(r,"Race Rank"), gv(r,"Comp. Rank"), gv(r,"PRB Rank")
                    
                    pure_r = fmt_int(gv(r, "Pure Rank"))
                    no_top = fmt_int(gv(r, "No. of Top"))
                    prim_r = fmt_int(gv(r, "Primary Rank"))
                    form_r = fmt_int(gv(r, "Form Rank"))
                    
                    html += '<tr>'
                    html += f'<td class="left-text"><b>{gv(r, "Horse")}</b></td>'
                    html += f'<td class="center-text">{f"{vp:.2f}" if isinstance(vp, float) else vp}</td><td class="center-text">{f"{pr:.2f}" if isinstance(pr, float) else pr}</td>'
                    html += f'<td class="center-text {rc(sr)}">{sr}</td><td class="center-text {rc(cr)}">{cr}</td><td class="center-text {rc(rr)}">{rr}</td><td class="center-text">{gv(r, "Race Rating", default=0)}</td><td class="center-text {rc(cpr)}">{cpr}</td><td class="center-text {rc(prb)}">{prb}</td>'
                    
                    if show_msai:
                        msai = fmt_int(gv(r, "MSAI Rank"))
                        html += f'<td class="center-text {rc(msai)}">{msai}</td>'
                        
                    html += f'<td class="center-text">{no_top}</td>'
                    html += f'<td class="center-text {rc(prim_r)}">{prim_r}</td><td class="center-text {rc(form_r)}">{form_r}</td>'
                    
                    ab = fmt_2dp(gv(r, "Ability"))
                    go = fmt_2dp(gv(r, "Going"))
                    di = fmt_2dp(gv(r, "Distance"))
                    cs = fmt_2dp(gv(r, "Course/Sim"))
                    tr = fmt_2dp(gv(r, "TrainrF"))
                    jo = fmt_2dp(gv(r, "JockyF"))
                    dr = fmt_2dp(gv(r, "Draw"))
                    sp = fmt_2dp(gv(r, "Speed"))
                    ts = fmt_2dp(gv(r, "Total"))
                    
                    html += f'<td class="center-text">{ab}</td><td class="center-text">{go}</td><td class="center-text">{di}</td><td class="center-text">{cs}</td><td class="center-text">{tr}</td><td class="center-text">{jo}</td><td class="center-text">{dr}</td><td class="center-text">{sp}</td>'
                    html += f'<td class="center-text" style="font-weight:bold;">{ts}</td>'
                    html += f'<td class="center-text {rc(pure_r)}" style="font-weight:bold;">{pure_r}</td>'
                    html += '</tr>'
                    
                html += "</tbody></table></div>"
                st.markdown(html, unsafe_allow_html=True)
                
            else:
                st.markdown("### Race Selection")
                courses = sorted([str(x).strip() for x in ta_df['Course'].dropna().unique() if str(x).strip()])
                
                for course in courses:
                    st.markdown(f"<div style='border-left: 6px solid #1a3a5f; padding-left: 12px; margin-top: 15px; margin-bottom: 10px; font-weight: bold; font-size: 16px; color: #1a3a5f; text-transform: uppercase;'>{course}</div>", unsafe_allow_html=True)
                    
                    c_df = ta_df[ta_df['Course'] == course]
                    races = c_df[['Time', 'Race Type', 'H/Cap']].drop_duplicates().sort_values('Time')
                    
                    cols = st.columns(10)
                    for idx, (_, r_row) in enumerate(races.iterrows()):
                        r_time = str(r_row['Time']).strip()
                        r_type = str(r_row['Race Type']).strip()
                        r_hcap = "Hcap" if str(r_row['H/Cap']).strip() == 'Y' else "Non-Hcap"
                        
                        btn_text = f"{r_time}\n{r_type} | {r_hcap}"
                        
                        if cols[idx % 10].button(btn_text, key=f"nav_{course}_{r_time}", use_container_width=True):
                            st.session_state.analysis_race = {'course': course, 'time': r_time}
                            st.rerun()

        else:
            st.info("No data available for today's races.")
