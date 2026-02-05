import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
import feedparser
from textblob import TextBlob
from datetime import datetime

# --- 1. 页面配置 ---
st.set_page_config(page_title="AI 量化实战终端 (Pro Global)", layout="wide", page_icon="⚡")
st.title("⚡ Crypto AI 终极实战终端 (Global Pro)")

# --- 2. 侧边栏配置 ---
with st.sidebar:
    st.header("🎛️ 策略控制台")
    
    # 基础参数
    st.subheader("1. 标的设置")
    default_symbol = 'BTC-USD'
    symbol_input = st.text_input("交易对 (Yahoo格式)", value=default_symbol).upper()
    tf_display = st.selectbox("时间周期", ['1h', '1d', '1wk'], index=0)
    limit = st.slider("K线样本数", 100, 500, 200)
    
    # 高级参数
    st.subheader("2. 数据源配置")
    etf_ticker = st.text_input("美股 ETF 代码", value='IBIT')
    
    # 权重配置
    st.subheader("3. 决策权重")
    w_tech = st.slider("技术面权重", 0.0, 1.0, 0.4)
    w_fund = st.slider("资金面权重", 0.0, 1.0, 0.3)
    w_onchain = st.slider("主力面权重", 0.0, 1.0, 0.2)
    w_news = st.slider("消息面权重", 0.0, 1.0, 0.1)

    if st.button('🚀 生成交易计划', type="primary"):
        st.rerun()

# --- 3. 核心分析引擎 (yfinance版) ---

class QuantEngine:
    def __init__(self, symbol, etf, tf, limit):
        self.symbol = symbol
        self.etf = etf
        self.tf = tf
        self.limit = limit

    def get_tech_analysis(self):
        """技术面深度分析"""
        try:
            df = yf.download(self.symbol, period="1mo", interval=self.tf, progress=False)
            if df.empty:
                st.error(f"❌ 无法获取 {self.symbol}")
                return None, 50, 0, 0
            
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = df.rename(columns={'Open': 'o', 'High': 'h', 'Low': 'l', 'Close': 'c', 'Volume': 'v'})
            df['ts'] = df.index
            
            current_price = df['c'].iloc[-1]
            
            # RSI
            rsi = ta.momentum.RSIIndicator(df['c']).rsi().iloc[-1]
            if rsi < 30: rsi_score = 90
            elif rsi > 70: rsi_score = 10
            else: rsi_score = 50 + (50 - rsi)
            
            # MACD
            macd = ta.trend.MACD(df['c'])
            hist = macd.macd_diff().iloc[-1]
            prev_hist = macd.macd_diff().iloc[-2] if len(df) > 1 else 0
            if hist > 0: macd_score = 90 if hist > prev_hist else 60
            else: macd_score = 10 if hist < prev_hist else 40
            
            # KDJ
            stoch = ta.momentum.StochasticOscillator(df['h'], df['l'], df['c'])
            k = stoch.stoch().iloc[-1]
            d = stoch.stoch_signal().iloc[-1]
            j = 3 * k - 2 * d
            kdj_score = 50
            if j < 20: kdj_score = 85
            elif j > 80: kdj_score = 15
            elif k > d: kdj_score = 65
            
            # EMA
            ema200 = ta.trend.EMAIndicator(df['c'], window=200).ema_indicator().iloc[-1]
            if pd.isna(ema200): ema200 = df['c'].mean()
            trend_score = 80 if current_price > ema200 else 20
            
            # ATR
            atr = ta.volatility.AverageTrueRange(df['h'], df['l'], df['c'], window=14).average_true_range().iloc[-1]
            
            final_tech_score = (rsi_score*0.2 + macd_score*0.3 + kdj_score*0.2 + trend_score*0.3)
            return df, final_tech_score, atr, ema200
        except Exception as e:
            return None, 50, 0, 0

    def get_etf_score(self):
        try:
            ticker = yf.Ticker(self.etf)
            df = ticker.history(period="1mo")
            if df.empty: return 50
            change = (df['Close'].iloc[-1] - df['Close'].iloc[-2])
            vol_ratio = df['Volume'].iloc[-1] / df['Volume'].mean()
            score = 50 + (20 * vol_ratio if change > 0 else -20 * vol_ratio)
            return max(0, min(100, score))
        except: return 50

    def get_money_flow_score(self, df):
        try:
            if df is None or df.empty: return 50
            cmf = ta.volume.ChaikinMoneyFlowIndicator(df['h'], df['l'], df['c'], df['v'], window=20).chaikin_money_flow().iloc[-1]
            score = 50 + (cmf * 150) 
            return max(0, min(100, score))
        except: return 50

    def get_news_score(self):
        try:
            keyword = self.symbol.split('-')[0]
            if keyword == 'BTC': keyword = 'Bitcoin'
            rss = f"https://news.google.com/rss/search?q={keyword}+crypto+when:1d&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss)
            scores = [TextBlob(e.title).sentiment.polarity for e in feed.entries[:10]]
            if not scores: return 50
            avg = sum(scores) / len(scores)
            return (avg + 1) * 50 
        except: return 50

# --- 4. 执行逻辑 ---

bot = QuantEngine(symbol_input, etf_ticker, tf_display, limit)

with st.spinner('AI 正在连接全球节点进行全域分析...'):
    df_k, s_tech, atr_val, ema_val = bot.get_tech_analysis()
    s_etf = bot.get_etf_score()
    s_whale = bot.get_money_flow_score(df_k)
    s_news = bot.get_news_score()

final_score = (s_tech * w_tech) + (s_etf * w_fund) + (s_whale * w_onchain) + (s_news * w_news)
current_price = df_k['c'].iloc[-1] if df_k is not None else 0

# --- 5. 界面展示 ---

st.subheader("🤖 AI 决策报告")

col_gauge, col_plan = st.columns([1.5, 2])

with col_gauge:
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = final_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "多空胜率评分", 'font': {'size': 20}},
        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "white"},
                 'steps': [{'range': [0, 40], 'color': '#FF4B4B'},
                           {'range': [40, 60], 'color': '#808080'},
                           {'range': [60, 100], 'color': '#00CC96'}]}
    ))
    fig.update_layout(height=300, margin=dict(l=20,r=20,t=30,b=20), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    
    status, color = ("🚀 极度看多", "green") if final_score >= 80 else \
                    (("🟢 谨慎做多", "green") if final_score >= 60 else \
                    (("📉 极度看空", "red") if final_score <= 20 else \
                    (("🔴 谨慎做空", "red") if final_score <= 40 else \
                    ("⚖️ 震荡观望", "gray"))))
    st.markdown(f"<h3 style='text-align: center; color: {color};'>{status}</h3>", unsafe_allow_html=True)

with col_plan:
    st.markdown("### 🎯 智能交易计划 (ATR动态风控)")
    if current_price > 0:
        if final_score >= 50:
            sl = current_price - (atr_val * 2)
            tp = current_price + (atr_val * 3)
        else:
            sl = current_price + (atr_val * 2)
            tp = current_price - (atr_val * 3)
            
        p1, p2, p3 = st.columns(3)
        p1.metric("入场价", f"${current_price:.2f}")
        p2.metric("止损价 (SL)", f"${sl:.2f}", delta_color="inverse")
        p3.metric("止盈价 (TP)", f"${tp:.2f}")
        
        # --- 🔥 找回的信心度模块 ---
        st.markdown("---")
        # 计算信心度：分数越偏离50，信心越足
        confidence = abs(final_score - 50) * 2 
        
        # 计算建议仓位
        pos_size = "0%"
        pos_desc = "空仓观望"
        pos_color = "gray"
        
        if confidence > 80: 
            pos_size = "20% (重仓)"
            pos_desc = "趋势极强，可激进建仓"
            pos_color = "red" # 醒目
        elif confidence > 40: 
            pos_size = "10% (标配)"
            pos_desc = "趋势形成，标准仓位"
            pos_color = "orange"
        
        # 显示
        c_col1, c_col2 = st.columns([1, 2])
        c_col1.metric("AI 信心度", f"{confidence:.0f}%")
        c_col2.markdown(f"**💰 建议仓位:** :{pos_color}[`{pos_size}`]")
        st.caption(f"💡 理由: {pos_desc} | 当前波动率(ATR): {atr_val:.2f}")
        
    else:
        st.warning("等待数据加载...")

st.markdown("---")

# 因子拆解
with st.expander("🔎 因子得分拆解", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("技术面", f"{s_tech:.0f}")
    c2.metric("资金面", f"{s_etf:.0f}")
    c3.metric("主力面 (CMF)", f"{s_whale:.0f}")
    c4.metric("消息面", f"{s_news:.0f}")

# 趋势图
st.subheader("📈 趋势确认")
if df_k is not None and not df_k.empty:
    fig_k = go.Figure()
    fig_k.add_trace(go.Candlestick(x=df_k['ts'], open=df_k['o'], high=df_k['h'], low=df_k['l'], close=df_k['c'], name='K线'))
    ema_list = [ema_val] * len(df_k)
    fig_k.add_trace(go.Scatter(x=df_k['ts'], y=ema_list, line=dict(color='orange'), name='EMA200基准线'))
    fig_k.update_layout(height=400, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_k, use_container_width=True)
else:
    st.info("暂无数据显示")
