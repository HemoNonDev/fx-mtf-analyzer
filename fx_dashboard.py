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
# 辞書型（Dictionary）というデータ構造を使い、「画面に表示する名前（例：USD/JPY (ドル円)）」と「Yahoo! Financeで使うコード（例：JPY=X）」をペアにして整理しています。

# --- サイドバーの設定 ---
selected_pair_name = st.sidebar.selectbox("表示する通貨ペアを選択", list(ticker_dict.keys()))
selected_symbol = ticker_dict[selected_pair_name]

# 最新レート表示用のプレースホルダーをサイドバーに作成
rate_placeholder = st.sidebar.empty() # 後からデータを上書きしたり、画面の一番上に結果を表示させたりしたい時のための『枠取り（予約）

st.sidebar.markdown("---")
st.sidebar.header("🔧 設定パネル")

# 💡 表記修正：「表示期間の設定」
st.sidebar.subheader("📅 表示期間の設定")
days_daily = st.sidebar.slider("日足の表示期間（日数）", min_value=30, max_value=365, value=240, step=10)
days_4h = st.sidebar.slider("4時間足の表示期間（日数）", min_value=15, max_value=180, value=30, step=5)

# 💡 表記修正：「水平線の設定」
st.sidebar.subheader("🎯 水平線の設定")
pips_range = st.sidebar.number_input("価格帯の幅（pips）", min_value=5, max_value=50, value=10)
min_touches = st.sidebar.number_input("最小反発回数（点数）", min_value=2, max_value=10, value=5)

# 💡 [新規追加] チャート縦幅の調整用スライダー
st.sidebar.subheader("📐 チャート画面の設定")
chart_height = st.sidebar.slider("チャートの縦幅（px）", min_value=300, max_value=800, value=450, step=25)

st.sidebar.markdown("---")
st.sidebar.caption("※紫太線：ロールリバーサル / 赤破線：抵抗線 / 青破線：支持線")

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
    
    # --- 日足の加工（初心者向け修正案） ---
    if not df_daily.empty:
        # 1. 時差情報（タイムゾーン）が付いていたら外して日付を揃える
        if df_daily.index.tz is not None:
            df_daily.index = df_daily.index.tz_localize(None)
        
        # 2. 移動平均線の計算（20MA, 100MA）
        # pandasのデータフレームから直接「終値（Close）」を使って計算する
        df_daily['MA20'] = df_daily['Close'].rolling(window=20).mean()
        df_daily['MA100'] = df_daily['Close'].rolling(window=100).mean()  # 週足20MA相当
        
        # 3. MA計算用に多めに取っていた過去データを、本来表示したい日数分に切り落とす
        start_cut_d = datetime.today() - timedelta(days=d_daily)
        df_daily = df_daily.loc[df_daily.index >= start_cut_d]

    # --- 4時間足の加工（初心者向け修正案） ---
    if not df_4h.empty:
        # 1. 時差情報（タイムゾーン）が付いていたら外して日付を揃える
        if df_4h.index.tz is not None:
            df_4h.index = df_4h.index.tz_localize(None)
        
        # 2. 移動平均線の計算（20MA, 120MA）
        # 'Close' 列から直接計算する
        df_4h['MA20'] = df_4h['Close'].rolling(window=20).mean()
        df_4h['MA120'] = df_4h['Close'].rolling(window=120).mean()  # 日足20MA相当
        
        # 3. MA計算用に多めに取っていた過去データを、本来表示したい日数分に切り落とす
        start_cut_4h = datetime.today() - timedelta(days=d_4h)
        df_4h = df_4h.loc[df_4h.index >= start_cut_4h]

    return df_4h, df_daily

# 水平線アルゴリズム
# --- 初心者向け修正案（アルゴリズム前半） ---
def find_advanced_lines(df, symbol_name, pips_window=10, min_touch=5):
    # .squeeze() を使わずに直接列を取り出す
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
    
    # 1pipの単位判定（普通の if / else に変更）
    if "JPY" in symbol_name:
        pip_unit = 0.01
    else:
        pip_unit = 0.0001
        
    tol = (pips_window / 2.0) * pip_unit
    hp, lp = np.array(hp), np.array(lp)
    lines_info = []
    
    # 高値と安値の配列を結合（普通の if / else に変更）
    if len(hp) > 0 and len(lp) > 0:
        all_pts = np.concatenate([hp, lp])
    else:
        all_pts = np.array([])
    
    # --- 初心者向け修正案（アルゴリズム後半） ---
    for pr in all_pts:
        # 1. 基準価格(pr)の近くにある「高値」と「安値」を集めるリスト
        near_hp = []
        for x in hp:
            if abs(x - pr) <= tol:
                near_hp.append(x)
                
        near_lp = []
        for x in lp:
            if abs(x - pr) <= tol:
                near_lp.append(x)
        
        h_cnt = len(near_hp)  # 高値の反発回数
        l_cnt = len(near_lp)  # 安値の反発回数
        
        # 2. 集まった価格の平均（中心価格）を計算する
        all_near_prices = near_hp + near_lp
        if len(all_near_prices) == 0:
            continue
        avg_pr = sum(all_near_prices) / len(all_near_prices)
        
        # 3. 重複チェック（すでに似たような水平線があればスキップ）
        is_duplicate = False
        for line in lines_info:
            if abs(line['price'] - avg_pr) < tol:
                is_duplicate = True
                break
        if is_duplicate:
            continue
            
        # 4. 条件判定とラインの登録
        # パターンA：ロールリバーサル（高値1回以上＋安値1回以上＋合計3回以上）➔ 紫の実線
        if h_cnt >= 1 and l_cnt >= 1 and (h_cnt + l_cnt) >= 3:
            lines_info.append({'price': avg_pr, 'color': '#c678dd', 'dash': 'solid', 'width': 2.5})
            
        # パターンB：レジスタンス線（高値で5回以上反発）➔ 赤の破線
        elif h_cnt >= min_touch:
            lines_info.append({'price': avg_pr, 'color': '#ff6c6b', 'dash': 'dash', 'width': 1.5})
            
        # パターンC：サポート線（安値で5回以上反発）➔ 青の破線
        elif l_cnt >= min_touch:
            lines_info.append({'price': avg_pr, 'color': '#51afef', 'dash': 'dash', 'width': 1.5})
            
    return lines_info

# チャート作成関数（初心者向け修正案）
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
    
    # 移動平均線（MA）の描画
    if is_daily:
        # 【日足チャートの場合】
        # 1. 日足 20MA（オレンジ色の細線）
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='日足 20MA', line=dict(color='#ff9800', width=1.2)))
        # 2. 週足 20MA相当の100MA（青色の太線）
        fig.add_trace(go.Scatter(x=df.index, y=df['MA100'], mode='lines', name='週足 20MA相当', line=dict(color='#1976d2', width=2.0)))
    else:
        # 【4時間足チャートの場合】
        # 1. 4時間足 20MA（オレンジ色の細線）
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='4H 20MA', line=dict(color='#ff9800', width=1.2)))
        # 2. 日足 20MA相当の120MA（青色の太線）
        fig.add_trace(go.Scatter(x=df.index, y=df['MA120'], mode='lines', name='日足 20MA相当', line=dict(color='#1976d2', width=2.0)))
    
    # 水平線を描画
    lines = find_advanced_lines(df, symbol_name, pips_win, min_t)
    for l in lines:
        fig.add_hline(y=l['price'], line_dash=l['dash'], line_color=l['color'], line_width=l['width'], opacity=0.7)
    
    # 左上の "Daily" / "4H" ラベル
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=1.12,
        text=f"<b>{label_text}</b>",
        showarrow=False,
        font=dict(size=26, color="#ffffff"),
        align="left"
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=height, # 💡 指定された高さ(height)をここで反映
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.85),
        xaxis=dict(showgrid=True, rangeslider=dict(visible=False)),
        yaxis=dict(showgrid=True)
    )
    return fig

# データの読み込み（くるくるローディング表示）
with st.spinner(f"{selected_pair_name} のデータを解析中..."):
    df_4h, df_daily = load_fx_data_mtf_separated(selected_symbol, days_daily, days_4h)

# 最新レートをサイドバーの通貨ペア選択直下に表示
if not df_4h.empty:
    latest = float(df_4h['Close'].values.flatten()[-1])
    rate_placeholder.metric(label="最新レート", value=f"{latest:.3f}")

# --- メイン画面描画（上下2段固定） ---
# 1. 日足チャートの描画
if not df_daily.empty:
    fig_daily = create_plotly_chart(df_daily, True, selected_pair_name, pips_range, min_touches, label_text="Daily", height=chart_height)
    st.plotly_chart(fig_daily, use_container_width=True)

# 2. 4時間足チャートの描画
if not df_4h.empty:
    fig_4h = create_plotly_chart(df_4h, False, selected_pair_name, pips_range, min_touches, label_text="4H", height=chart_height)
    st.plotly_chart(fig_4h, use_container_width=True)
