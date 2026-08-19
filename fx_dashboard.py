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
st.sidebar.caption("※日足紫線：日足ロールリバーサル / 4Hピンク線：4Hロールリバーサル / 赤破線：抵抗線 / 青破線：支持線")

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

# 水平線アルゴリズム
def find_advanced_lines(df, symbol_name, pips_window=10, min_touch=5, rr_color='#c678dd'):
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
    
    # # 1pipの単位判定
    # if "JPY" in symbol_name:
    #     pip_unit = 0.01
    # else:
    #     pip_unit = 0.0001
        
    # tol = (pips_window / 2.0) * pip_unit
    # hp, lp = np.array(hp), np.array(lp)
    # lines_info = []
    
    # # 高値と安値の配列を結合
    # if len(hp) > 0 and len(lp) > 0:
    #     all_pts = np.concatenate([hp, lp])
    # else:
    #     all_pts = np.array([])
    
    # for pr in all_pts:
    #     near_hp = [x for x in hp if abs(x - pr) <= tol]
    #     near_lp = [x for x in lp if abs(x - pr) <= tol]
        
    #     h_cnt = len(near_hp)
    #     l_cnt = len(near_lp)
        
    #     all_near_prices = near_hp + near_lp
    #     if len(all_near_prices) == 0:
    #         continue
    #     avg_pr = sum(all_near_prices) / len(all_near_prices)
        
    #     # 重複チェック
    #     is_duplicate = False
    #     for line in lines_info:
    #         if abs(line['price'] - avg_pr) < tol:
    #             is_duplicate = True
    #             break
    #     if is_duplicate:
    #         continue

    # 1pipの単位判定
    if "JPY" in symbol_name:
        pip_unit = 0.01
    else:
        pip_unit = 0.0001
        
    half_width = (pips_window / 2.0) * pip_unit  # ±5pips相当の幅
    hp, lp = np.array(hp), np.array(lp)
    zones_info = []  # 登録用のリスト（lines_infoから名称変更）
    
    # 高値と安値の配列を結合
    if len(hp) > 0 and len(lp) > 0:
        all_pts = np.concatenate([hp, lp])
    else:
        all_pts = np.array([])
    
    for center_pr in all_pts:
        # 帯（ゾーン）の下限価格(y0)と上限価格(y1)を設定
        y0 = center_pr - half_width
        y1 = center_pr + half_width
        
        # 1. 帯の上側（サポート）で安値が反応した回数
        support_count = np.sum((lp >= y0) & (lp <= y1))
        
        # 2. 帯の下側（レジスタンス）で高値が反応した回数
        resistance_count = np.sum((hp >= y0) & (hp <= y1))
        
        # 【条件】サポート3回以上 かつ レジスタンス3回以上（回数は任意）
        if support_count >= 3 and resistance_count >= 3:
            
            # 重複チェック（すでに登録済みの帯と中心価格が近くないか）
            is_duplicate = False
            for zone in zones_info:
                if abs(zone['center'] - center_pr) < (pips_window * pip_unit):
                    is_duplicate = True
                    break
            
            if is_duplicate:
                continue
                
            # 条件を満たした帯（ゾーン）情報を追加
            zones_info.append({
                'y0': y0,
                'y1': y1,
                'center': center_pr,
                'support_count': support_count,
                'resistance_count': resistance_count
            })
            
    #     # パターンA：ロールリバーサル（時間足に応じた指定色を適用）
    #     if h_cnt >= 1 and l_cnt >= 1 and (h_cnt + l_cnt) >= 3:
    #         lines_info.append({'price': avg_pr, 'color': rr_color, 'dash': 'solid', 'width': 2.5})
            
    #     # パターンB：レジスタンス線（赤の破線）
    #     elif h_cnt >= min_touch:
    #         lines_info.append({'price': avg_pr, 'color': '#ff6c6b', 'dash': 'dash', 'width': 1.5})
            
    #     # パターンC：サポート線（青の破線）
    #     elif l_cnt >= min_touch:
    #         lines_info.append({'price': avg_pr, 'color': '#51afef', 'dash': 'dash', 'width': 1.5})
            
    # return lines_info

        # --- ここから条件分岐と登録処理 ---
        
        # パターンA：ロールリバーサル帯（サポート・レジスタンス両方で反発）
        if support_count >= 1 and resistance_count >= 1 and (support_count + resistance_count) >= 3:
            zones_info.append({
                'y0': y0,
                'y1': y1,
                'center': center_pr,
                'fillcolor': 'rgba(180, 100, 255, 0.25)',  # ロールリバーサル用の指定色（半透明）
                'type': 'roll_reversal'
            })
            
        # パターンB：レジスタンス帯（赤系の半透明）
        elif resistance_count >= min_touch:
            zones_info.append({
                'y0': y0,
                'y1': y1,
                'center': center_pr,
                'fillcolor': 'rgba(255, 108, 107, 0.2)',  # #ff6c6bの半透明
                'type': 'resistance'
            })
            
        # パターンC：サポート帯（青系の半透明）
        elif support_count >= min_touch:
            zones_info.append({
                'y0': y0,
                'y1': y1,
                'center': center_pr,
                'fillcolor': 'rgba(81, 175, 239, 0.2)',  # #51afefの半透明
                'type': 'support'
            })
            
    return zones_info

# チャート作成関数
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
    
    # 移動平均線（MA）の描画とロールリバーサル色の分岐
    if is_daily:
        # 日足チャート：日足20MA（オレンジ）、週足20MA相当（青）
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='日足 20MA', line=dict(color='#ff9800', width=1.5)))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA100'], mode='lines', name='週足 20MA相当', line=dict(color='#1976d2', width=2.0)))
        rr_color = '#c678dd'  # 日足用ロールリバーサル（紫色）
    else:
        # 4時間足チャート：4H20MA（緑）、日足20MA相当（120MA：オレンジ）
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='4H 20MA', line=dict(color='#4caf50', width=1.5)))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA120'], mode='lines', name='日足 20MA相当', line=dict(color='#ff9800', width=2.0)))
        rr_color = '#e06c75'  # 4時間足用ロールリバーサル（ピンク/マゼンタ色）
    
    # 水平線を描画
    lines = find_advanced_lines(df, symbol_name, pips_win, min_t, rr_color=rr_color)
    # Plotlyへの描画処理
    for zone in zones_info:
        fig.add_hrect(
            y0=zone["y0"],
            y1=zone["y1"],
            fillcolor=zone["fillcolor"],  # 条件分岐で設定した色を動的に適用
            line_width=0,                 # 枠線なし
            layer="below",                # ローソク足の裏に配置
        )
    
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
        height=height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.85),
        xaxis=dict(showgrid=True, rangeslider=dict(visible=False)),
        yaxis=dict(showgrid=True)
    )
    return fig

# データの読み込み
with st.spinner(f"{selected_pair_name} のデータを解析中..."):
    df_4h, df_daily = load_fx_data_mtf_separated(selected_symbol, days_daily, days_4h)

# 最新レートをサイドバーに表示（桁数を自動判定）
if not df_4h.empty:
    latest = float(df_4h['Close'].iloc[-1])
    digits = 3 if "JPY" in selected_symbol else 5
    rate_placeholder.metric(label="最新レート", value=f"{latest:.{digits}f}")

# --- メイン画面描画（上下2段固定） ---
if not df_daily.empty:
    fig_daily = create_plotly_chart(df_daily, True, selected_pair_name, pips_range, min_touches, label_text="Daily", height=chart_height)
    st.plotly_chart(fig_daily, use_container_width=True)

if not df_4h.empty:
    fig_4h = create_plotly_chart(df_4h, False, selected_pair_name, pips_range, min_touches, label_text="4H", height=chart_height)
    st.plotly_chart(fig_4h, use_container_width=True)
