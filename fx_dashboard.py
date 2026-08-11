import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 1. ページの設定（ワイドレイアウト）
st.set_page_config(layout="wide", page_title="FX MTF Dual Chart Dashboard")

st.title("📊 FX Interactive MTF Dashboard")

# 通貨ペアの選択リスト
ticker_dict = {
    "USD/JPY (ドル円)": "JPY=X",
    "EUR/JPY (ユーロ円)": "EURJPY=X",
    "GBP/JPY (ポンド円)": "GBPJPY=X",
    "EUR/USD (ユーロドル)": "EUR=X",
    "GBP/USD (ポンドドル)": "GBPUSD=X",
    "AUD/JPY (豪ドル円)": "AUDJPY=X",
    "NZD/JPY (キウイ円)": "NZDJPY=X",
    "CAD/JPY (カナダ円)": "CADJPY=X",
    "CHF/JPY (フラン円)": "CHFJPY=X"
}

# --- サイドバーの設定 ---
selected_pair_name = st.sidebar.selectbox("表示する通貨ペアを選択", list(ticker_dict.keys()))
selected_symbol = ticker_dict[selected_pair_name]

# 最新レート表示用のプレースホルダーをサイドバーに作成
rate_placeholder = st.sidebar.empty()

st.sidebar.markdown("---")
st.sidebar.header("🔧 設定パネル")

st.sidebar.subheader("📅 表示期間の設定")
days_daily = st.sidebar.slider("日足の表示期間（日数）", min_value=30, max_value=365, value=240, step=10)
days_4h = st.sidebar.slider("4時間足の表示期間（日数）", min_value=15, max_value=180, value=30, step=5)

st.sidebar.subheader("🎯 水平線の設定")
pips_range = st.sidebar.number_input("価格帯の幅（pips）", min_value=5, max_value=50, value=10)
min_touches = st.sidebar.number_input("最小反発回数（点数）", min_value=2, max_value=10, value=5)

st.sidebar.subheader("📐 チャート画面の設定")
chart_height = st.sidebar.slider("チャートの縦幅（px）", min_value=300, max_value=800, value=450, step=25)

st.sidebar.markdown("---")
st.sidebar.caption("※日足水平線：紫太線/赤破線/青破線  ※4H足水平線：ピンク太線/オレンジ破線/水色破線")

# --- データ取得・計算関数 ---
@st.cache_data
def load_fx_data_mtf_separated(symbol, d_daily, d_4h):
    end_date = datetime.today()
    
    # 1. 日足データ取得
    start_date_daily = end_date - timedelta(days=d_daily + 120)
    df_daily = yf.download(symbol, start=start_date_daily, end=end_date, interval="1d", progress=False)
    
    # 2. 4時間足データ取得
    start_date_4h = end_date - timedelta(days=d_4h + 35)
    df_4h = yf.download(symbol, start=start_date_4h, end=end_date, interval="4h", progress=False)
    
    # 2重構造（MultiIndex）になっている列名を1重のシンプルな列名に平坦化する
    if isinstance(df_daily.columns, pd.MultiIndex):
        df_daily.columns = df_daily.columns.get_level_values(0)
    if isinstance(df_4h.columns, pd.MultiIndex):
        df_4h.columns = df_4h.columns.get_level_values(0)

    # --- 日足の加工 ---
    if not df_daily.empty:
        if df_daily.index.tz is not None:
            df_daily.index = df_daily.index.tz_localize(None)
        
        df_daily['MA20'] = df_daily['Close'].rolling(window=20).mean()
        df_daily['MA100'] = df_daily['Close'].rolling(window=100).mean()  # 週足20MA相当
        
        start_cut_d = datetime.today() - timedelta(days=d_daily)
        df_daily = df_daily.loc[df_daily.index >= start_cut_d]

    # --- 4時間足の加工 ---
    if not df_4h.empty:
        if df_4h.index.tz is not None:
            df_4h.index = df_4h.index.tz_localize(None)
        
        df_4h['MA20'] = df_4h['Close'].rolling(window=20).mean()
        df_4h['MA120'] = df_4h['Close'].rolling(window=120).mean()  # 日足20MA相当
        
        start_cut_4h = datetime.today() - timedelta(days=d_4h)
        df_4h = df_4h.loc[df_4h.index >= start_cut_4h]

    return df_4h, df_daily

# 水平線アルゴリズム（解説通りforループ処理）
def find_advanced_lines(df, symbol_name, pips_window=10, min_touch=5):
    highs = df['High']
    lows = df['Low']
    hp, lp = [], []
    
    # ピボット（山・谷）の抽出
    for i in range(3, len(df)-3):
        # 高値の山（ピボットハイ）判定
        if (highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i-2] and highs.iloc[i] > highs.iloc[i-3] and
            highs.iloc[i] > highs.iloc[i+1] and highs.iloc[i] > highs.iloc[i+2] and highs.iloc[i] > highs.iloc[i+3]):
            hp.append(highs.iloc[i])
            
        # 安値の谷（ピボットロー）判定
        if (lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i-2] and lows.iloc[i] < lows.iloc[i-3] and
            lows.iloc[i] < lows.iloc[i+1] and lows.iloc[i] < lows.iloc[i+2] and lows.iloc[i] < lows.iloc[i+3]):
            lp.append(lows.iloc[i])
    
    # 1pipの単位判定
    if "JPY" in symbol_name:
        pip_unit = 0.01
    else:
        pip_unit = 0.0001
        
    tol = (pips_window / 2.0) * pip_unit
    hp, lp = np.array(hp), np.array(lp)
    lines_info = []
    
    # 高値と安値の配列を結合
    if len(hp) > 0 and len(lp) > 0:
        all_pts = np.concatenate([hp, lp])
    else:
        all_pts = np.array([])
    
    for pr in all_pts:
        near_hp = []
        for x in hp:
            if abs(x - pr) <= tol:
                near_hp.append(x)
                
        near_lp = []
        for x in lp:
            if abs(x - pr) <= tol:
                near_lp.append(x)
        
        h_cnt = len(near_hp)
        l_cnt = len(near_lp)
        
        all_near_prices = near_hp + near_lp
        if len(all_near_prices) == 0:
            continue
        avg_pr = sum(all_near_prices) / len(all_near_prices)
        
        is_duplicate = False
        for line in lines_info:
            if abs(line['price'] - avg_pr) < tol:
                is_duplicate = True
                break
        if is_duplicate:
            continue
            
        # 種別（'rr', 'res', 'sup'）判定のみを行う
        if h_cnt >= 1 and l_cnt >= 1 and (h_cnt + l_cnt) >= 3:
            lines_info.append({'price': avg_pr, 'type': 'rr'})
        elif h_cnt >= min_touch:
            lines_info.append({'price': avg_pr, 'type': 'res'})
        elif l_cnt >= min_touch:
            lines_info.append({'price': avg_pr, 'type': 'sup'})
            
    return lines_info

# チャート作成関数（MAの共通色設定と水平線の色分けを実施）
def create_plotly_chart(df, is_daily, symbol_name, pips_win, min_t, label_text, height=450):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='#ef5350', # 陽線（赤）
        decreasing_line_color='#2196f3', # 陰線（青）
        showlegend=False
    )])
    
    # --- 移動平均線（MA）の描画 ---
    # 週足20MA：緑（#4caf50） / 日足20MA：青（#1976d2） / 4H20MA：オレンジ（#ff9800）
    if is_daily:
        # 1. 日足 20MA（青線）
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='日足 20MA', line=dict(color='#1976d2', width=1.5)))
        # 2. 週足 20MA相当の100MA（緑太線）
        fig.add_trace(go.Scatter(x=df.index, y=df['MA100'], mode='lines', name='週足 20MA相当', line=dict(color='#4caf50', width=2.0)))
    else:
        # 1. 4時間足 20MA（オレンジ線）
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='4H 20MA', line=dict(color='#ff9800', width=1.2)))
        # 2. 日足 20MA相当の120MA（青太線 ➔ 日足20MAと同色！）
        fig.add_trace(go.Scatter(x=df.index, y=df['MA120'], mode='lines', name='日足 20MA相当', line=dict(color='#1976d2', width=2.0)))
    
    # 水平線を描画（is_daily で色を分岐）
    lines = find_advanced_lines(df, symbol_name, pips_win, min_t)
    for l in lines:
        if is_daily:
            # 【日足用のカラー設定】
            if l['type'] == 'rr':
                color, dash, width = '#c678dd', 'solid', 2.5   # 紫太線
            elif l['type'] == 'res':
                color, dash, width = '#ff6c6b', 'dash', 1.5    # 赤破線
            else:
                color, dash, width = '#51afef', 'dash', 1.5    # 青破線
        else:
            # 【4時間足用のカラー設定】
            if l['type'] == 'rr':
                color, dash, width = '#e06c75', 'solid', 2.5   # ピンク太線
            elif l['type'] == 'res':
                color, dash, width = '#d19a66', 'dash', 1.5    # オレンジ破線
            else:
                color, dash, width = '#56b6c2', 'dash', 1.5    # 水色破線

        fig.add_hline(y=l['price'], line_dash=dash, line_color=color, line_width=width, opacity=0.7)
    
    # 左上の "Daily" / "4H" ラベル
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=1
