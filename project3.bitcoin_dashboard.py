import streamlit as st
import pyupbit
import yfinance as yf
import pandas as pd
import time
import numpy as np
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(page_title="통합 스캘핑 지휘소 v9.5", layout="wide")

# 스타일 설정
st.markdown("""
    <style>
    .stMetric { background-color: #1d2127; padding: 10px; border-radius: 10px; border-left: 5px solid #00ff00; }
    .box-info { padding:15px; border-radius:10px; background-color: #161b22; border: 1px solid #30363d; margin-bottom: 10px; }
    .progress-text { font-size: 14px; font-weight: bold; margin-bottom: 2px; display: flex; justify-content: space-between; }
    .near-analysis { padding: 10px; background-color: #0e1117; border-radius: 5px; border: 1px solid #444; margin-top: 10px; }
    .profit-zone { background-color: #003311; border: 2px solid #00ff00; color: #00ff00; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 10px; }
    .loss-zone { background-color: #330000; border: 2px solid #ff4b4b; color: #ff4b4b; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 10px; }
    .status-bar { padding: 12px; border-radius: 10px; background-color: #161b22; border: 1px solid #30363d; margin-bottom: 10px; display: flex; justify-content: space-around; align-items: center; }
    .market-badge { padding: 4px 12px; border-radius: 15px; font-size: 0.8rem; font-weight: bold; margin-left: 10px; }
    </style>
    """, unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────
# 1. 사이드바 설정
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🌐 통합 컨트롤러")
    market_select = st.radio("📡 분석 시장 선택", ["국내 코인", "미국 주식"])
    st.divider()
    
    ticker = st.text_input("대상 티커 (KRW-BTC 또는 TSLA)", value="KRW-BTC" if market_select == "국내 코인" else "TSLA").upper()
    
    st.write("### 💰 나의 포지션")
    my_buy_price = st.number_input("나의 매수 단가 (0이면 비활성)", value=0.0, step=0.1)
    target_profit = st.slider("목표 익절률 (%)", 0.5, 10.0, 2.0, step=0.1)
    stop_loss = st.slider("최대 손절률 (%)", 0.5, 10.0, 1.5, step=0.1)
    st.divider()
    
    box_sensitivity = st.slider("호가벽 감지 강도", 1.2, 5.0, 1.8)
    st.divider()
    lookback_period = st.slider("박스권 분석 기간 (분)", 10, 60, 30)
    range_x = st.slider("분석 범위 (x%)", 0.1, 10.0, 1.0, step=0.1)
    
    if market_select == "미국 주식":
        chart_view_limit = st.slider("차트 캔들 표시 개수", 10, 120, 40)
        
    is_confirmed = st.checkbox("✅ 실시간 감시 활성화")

# ─────────────────────────────────────────────────────────
# 공통 함수: 수익률 전광판
# ─────────────────────────────────────────────────────────
def display_profit_status(curr_price, buy_price, market_type="coin"):
    if buy_price > 0:
        unit = "원" if market_type == "coin" else "$"
        fmt = ",.0f" if market_type == "coin" else ",.2f"
        tp_price = buy_price * (1 + target_profit / 100)
        sl_price = buy_price * (1 - stop_loss / 100)
        profit_pct = ((curr_price - buy_price) / buy_price) * 100
        
        if profit_pct >= target_profit:
            st.markdown(f'<div class="profit-zone"><h3>🔥 목표가 도달! 익절 준비: {profit_pct:.2f}%</h3><p>현재가: {curr_price:{fmt}}{unit} (익절기준: {tp_price:{fmt}}{unit})</p></div>', unsafe_allow_html=True)
        elif profit_pct <= -stop_loss:
            st.markdown(f'<div class="loss-zone"><h3>🚨 손절가 도달! 즉시 탈출: {profit_pct:.2f}%</h3><p>현재가: {curr_price:{fmt}}{unit} (손절기준: {sl_price:{fmt}}{unit})</p></div>', unsafe_allow_html=True)
        else:
            p_color = "#00ff00" if profit_pct > 0 else "#ff4b4b"
            st.markdown(f"""
                <div class="status-bar">
                    <span style='font-size:1.1rem;'>현재 수익률: <b style='color:{p_color};'>{profit_pct:.2f}%</b></span>
                    <span style='color:#8b949e;'>|</span>
                    <span style='font-size:1rem;'>🎯 익절가: <b style='color:#00ff00;'>{tp_price:{fmt}}{unit}</b></span>
                    <span style='font-size:1rem;'>🛑 손절가: <b style='color:#ff4b4b;'>{sl_price:{fmt}}{unit}</b></span>
                </div>
            """, unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────
# 2. 미장 전용 실행 함수
# ─────────────────────────────────────────────────────────
def run_us_market():
    st.title(f"🇺🇸 미장 실시간 조종석: {ticker}")
    container = st.empty()
    while True:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(interval="1m", period="1d", prepost=True)
            
            if not df.empty:
                df = df.dropna(subset=['Close'])
                curr = df['Close'].iloc[-1]
                analysis_df = df.tail(lookback_period)
                upper = analysis_df['High'].max()
                lower = analysis_df['Low'].min()
                
                vol_avg = analysis_df['Volume'].mean()
                if pd.isna(vol_avg) or vol_avg == 0: vol_avg = 1.0
                
                pseudo_asks, pseudo_bids = [], []
                for i in range(1, 11):
                    a_price, b_price = curr * (1 + (i * 0.0005)), curr * (1 - (i * 0.0005))
                    a_size = vol_avg * np.random.uniform(0.5, 1.5) * (2.0 if a_price >= upper * 0.999 else 1.0)
                    b_size = vol_avg * np.random.uniform(0.5, 1.5) * (2.0 if b_price <= lower * 1.001 else 1.0)
                    pseudo_asks.append({'ask_price': a_price, 'ask_size': a_size})
                    pseudo_bids.append({'bid_price': b_price, 'bid_size': b_size})
                
                orderbook_pseudo = pd.concat([pd.DataFrame(pseudo_asks)[::-1], pd.DataFrame(pseudo_bids)], axis=1).reset_index(drop=True)
                
                info = stock.info
                state = info.get('marketState', 'UNKNOWN')
                state_map = {"PRE": ("장전(PRE)", "#ffaa00"), "REGULAR": ("장중(LIVE)", "#00ff00"), "POST": ("장후(POST)", "#5555ff"), "CLOSED": ("장마감", "#888888")}
                s_text, s_color = state_map.get(state, (state, "#ccc"))

                with container.container():
                    st.markdown(f"**상태:** <span class='market-badge' style='background-color:{s_color}; color:white;'>{s_text}</span>", unsafe_allow_html=True)
                    display_profit_status(curr, my_buy_price, "stock")
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("현재가", f"$ {curr:.2f}")
                    t_ask = sum([x['ask_size'] for x in pseudo_asks]); t_bid = sum([x['bid_size'] for x in pseudo_bids])
                    bid_ratio = (t_bid / (t_ask + t_bid)) * 100 if (t_ask + t_bid) > 0 else 50.0
                    m2.metric("전체 잔량 비중", f"{bid_ratio:.1f} %")
                    m3.metric("박스 상단", f"$ {upper:.2f}")
                    m4.metric("박스 하단", f"$ {lower:.2f}")
                    
                    st.divider()

                    # 캔들스틱 차트 및 ID (Time 기반 Key)
                    st.write(f"#### 🕯️ 실시간 캔들 차트 (최근 {chart_view_limit}분)")
                    chart_df = df.tail(chart_view_limit)
                    fig = go.Figure(data=[go.Candlestick(
                        x=chart_df.index,
                        open=chart_df['Open'], high=chart_df['High'],
                        low=chart_df['Low'], close=chart_df['Close'],
                        increasing_line_color='#00ff00', decreasing_line_color='#ff4b4b'
                    )])
                    
                    # 가격 범위 최적화 (Auto-Zoom)
                    margin = (chart_df['High'].max() - chart_df['Low'].min()) * 0.15
                    fig.update_layout(
                        template='plotly_dark', height=400, margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_rangeslider_visible=False,
                        yaxis=dict(range=[chart_df['Low'].min() - margin, chart_df['High'].max() + margin], gridcolor='#333')
                    )
                    
                    # time.time()을 사용하여 매 루프마다 고유한 key 생성
                    st.plotly_chart(fig, use_container_width=True, key=f"us_chart_{ticker}_{time.time()}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("#### 🧱 실시간 호가 (추정)")
                        st.dataframe(orderbook_pseudo.style.background_gradient(subset=['ask_size'], cmap='Reds').background_gradient(subset=['bid_size'], cmap='Greens'), use_container_width=True)
                    with col2:
                        st.write("#### 🤖 통합 분석")
                        st.markdown('<div class="box-info">', unsafe_allow_html=True)
                        if is_confirmed and curr < lower: st.error("🚨 즉시 대응! 박스 하단 붕괴")
                        elif is_confirmed and curr > upper: st.success("🚀 상단 돌파! 추세 추종")
                        else: st.info("⚖️ 박스권 내 안정적 흐름")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.progress(int(max(0, min(100, bid_ratio))))

            time.sleep(5)
        except Exception as e:
            st.error(f"오류: {e}"); time.sleep(10)
# ─────────────────────────────────────────────────────────
# 3. 코인용 함수들
# ─────────────────────────────────────────────────────────
def get_orderbook_data(ticker):
    try:
        orderbook = pyupbit.get_orderbook(ticker)
        df = pd.DataFrame(orderbook['orderbook_units'])
        return df, orderbook
    except: return None, None

def get_historical_resistance(ticker):
    try:
        df_hist = pyupbit.get_ohlcv(ticker, interval="minute1", count=200)
        vol_threshold = df_hist['volume'].mean() * 2.5
        resistance_zones = df_hist[df_hist['volume'] > vol_threshold]
        return sorted(resistance_zones['high'].unique(), reverse=True)
    except: return []

def calculate_dynamic_box(ticker, period):
    try:
        df = pyupbit.get_ohlcv(ticker, interval="minute1", count=period)
        return df['high'].max(), df['low'].min()
    except: return None, None
# ─────────────────────────────────────────────────────────
# 메인 실행 로직
# ─────────────────────────────────────────────────────────
if market_select == "미국 주식":
    run_us_market()
else:
    # 코인 파트 로직과 동일함
    st.title(f"🚀 {ticker} 통합 스캘핑 시스템")
    placeholder = st.empty()

    if 'last_ticker' not in st.session_state or st.session_state.last_ticker != ticker:
        st.session_state.hist_res = get_historical_resistance(ticker)
        st.session_state.last_ticker = ticker
        st.session_state.last_update = 0

    while True:
        if time.time() - st.session_state.last_update > 60:
            st.session_state.box_upper, st.session_state.box_lower = calculate_dynamic_box(ticker, lookback_period)
            st.session_state.last_update = time.time()

        df, raw_data = get_orderbook_data(ticker)
        
        if df is not None:
            with placeholder.container():
                curr_price = df['ask_price'].iloc[0]
                display_profit_status(curr_price, my_buy_price, "coin")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("현재가", f"{curr_price:,} 원")
                t_bid = raw_data['total_bid_size']
                t_ask = raw_data['total_ask_size']
                bid_ratio = (t_bid / (t_ask + t_bid)) * 100 if (t_ask + t_bid) > 0 else 50.0
                
                m2.metric("전체 잔량 비중", f"{bid_ratio:.1f} %")
                m3.metric("박스 상단", f"{st.session_state.box_upper:,}")
                m4.metric("박스 하단", f"{st.session_state.box_lower:,}")
                st.divider()
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.write("#### 🧱 실시간 호가 및 매수벽")
                    view_df = df.copy()[['ask_price', 'ask_size', 'bid_size', 'bid_price']]
                    st.dataframe(view_df.style.background_gradient(subset=['ask_size'], cmap='Reds')
                                             .background_gradient(subset=['bid_size'], cmap='Greens'), use_container_width=True)
                with col2:
                    st.write(f"#### 🎯 근접 호가 분석 (현재가 ±{range_x}%)")
                    upper_limit, lower_limit = curr_price * (1 + range_x/100), curr_price * (1 - range_x/100)
                    near_ask_total = df[df['ask_price'] <= upper_limit]['ask_size'].sum()
                    near_bid_total = df[df['bid_price'] >= lower_limit]['bid_size'].sum()
                    near_ratio = (near_bid_total / (near_ask_total + near_bid_total)) * 100 if (near_ask_total + near_bid_total) > 0 else 50.0
                    
                    st.markdown(f'<div class="near-analysis"><p>범위 내 매도 총량: <b>{near_ask_total:,.2f}</b></p><p>범위 내 매수 총량: <b>{near_bid_total:,.2f}</b></p><div style="display:flex; justify-content:space-between;"><span>실질 매수 비중</span><span style="color:{"#00ff00" if near_ratio >= 50 else "#ff4b4b"};">{near_ratio:.1f}%</span></div></div>', unsafe_allow_html=True)
                    st.divider()
                    st.write("#### 🤖 통합 분석 진단")
                    avg_bid, avg_ask = df['bid_size'].mean(), df['ask_size'].mean()
                    support_wall = df[df['bid_size'] > avg_bid * box_sensitivity]['bid_price'].min()
                    resistance_wall = df[df['ask_size'] > avg_ask * box_sensitivity]['ask_price'].max()
                    st.markdown('<div class="box-info">', unsafe_allow_html=True)
                    if not pd.isna(resistance_wall): st.error(f"⛔ 매도벽: {resistance_wall:,} 원")
                    if not pd.isna(support_wall): st.info(f"🛡️ 매수벽: {support_wall:,} 원")
                    if is_confirmed and curr_price < st.session_state.box_lower: st.error("🚨 즉시 손절! 박스 붕괴")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.divider()
                if bid_ratio >= 60: m_color, label = "#00ff00", "🟢 전체 시장: 매수 우세"
                elif bid_ratio >= 40: m_color, label = "#ffaa00", "🟡 전체 시장: 균형"
                else: m_color, label = "#ff4b4b", "🔴 전체 시장: 매도 압박"
                
                st.markdown(f'<div class="progress-text"><span style="color:{m_color};">{label}</span><span>{bid_ratio:.1f}%</span></div>', unsafe_allow_html=True)
                safe_progress_coin = int(max(0, min(100, bid_ratio))) if not pd.isna(bid_ratio) else 50
                st.progress(safe_progress_coin)
        time.sleep(0.5)