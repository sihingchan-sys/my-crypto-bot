import streamlit as st
import ccxt
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
import feedparser
from textblob import TextBlob
from datetime import datetime

# --- 1. 页面配置 ---
st.set_page_config(page_title="AI 量化实战终端 (Pro Max)", layout="wide", page_icon="⚡")
st.title("⚡ Crypto AI 终极实战终端 (Day 5 Pro Max)")

# --- 2. 侧边栏配置 ---
with st.sidebar:
    st.header("🎛️ 策略控制台")
    
    # 基础参数
    st.subheader("1. 标的设置")
    crypto_symbol = st.text_input("交易对", value='BTC/USDT').upper()
    timeframe = st.selectbox("时间周期", ['1h', '4h', '1d'], index=0)
    limit = st.slider("K线样本数", 100, 1000, 200)
    
    # 高级参数
    st.subheader("2. 数据源配置")
    etf_ticker = st.text_input("美股 ETF 代码", value='IBIT')
    whale_threshold = st.number_input("巨鲸阈值 ($)", value=100000, step=10000)
    
    # 权重配置
    st.subheader("3. 决策权重 (总和建议1.0)")
    w_tech = st.slider("技术面权重", 0.0, 1.0, 0.4, help="K线指标的占比")
    w_fund = st.slider("资金面权重", 0.0, 1.0, 0.3, help="ETF资金流向占比")
    w_onchain = st.slider("链上权重", 0.0, 1.0, 0.2, help="巨鲸大单占比")
    w_news = st.slider("消息面权重", 0.0, 1.0, 0.1, help="新闻情绪占比")

    if st.button('🚀 生成交易计划', type="primary"):
        st.rerun()

# --- 3. 核心分析引擎 ---

class QuantEngine:
    def __init__(self, symbol, etf, tf, limit):
        self.symbol = symbol
        self.etf = etf
        self.tf = tf
        self.limit = limit
        self.exchange = ccxt.binance({'enableRateLimit': True})

    def get_tech_analysis(self):
        """技术面深度分析 (5大指标)"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.tf, limit=self.limit)
            df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            current_price = df['c'].iloc[-1]
            
            # 1. RSI (强弱)
            rsi = ta.momentum.RSIIndicator(df['c']).rsi().iloc[-1]
            if rsi < 30: rsi_score = 90 # 超卖反弹
            elif rsi > 70: rsi_score = 10 # 超买回调
            else: rsi_score = 50 + (50 - rsi)
            
            # 2. MACD (趋势动能)
            macd = ta.trend.MACD(df['c'])
            hist = macd.macd_diff().iloc[-1]
            prev_hist = macd.macd_diff().iloc[-2]
            # 柱状图变长(动能强)得分高，变短得分低
            if hist > 0: macd_score = 90 if hist > prev_hist else 60
            else: macd_score = 10 if hist < prev_hist else 40
            
            # 3. KDJ (短线敏感) - 需要手动计算 J 线
            stoch = ta.momentum.StochasticOscillator(df['h'], df['l'], df['c'])
            k = stoch.stoch().iloc[-1]
            d = stoch.stoch_signal().iloc[-1]
            j = 3 * k - 2 * d
            # J线金叉K线(J上穿K)看多
            kdj_score = 50
            if j < 20 and j > df['c'].pct_change().iloc[-1]: kdj_score = 85 # 底部拐头
            elif j > 80: kdj_score = 15 # 顶部钝化
            elif k > d: kdj_score = 65 # 金叉状态
            
            # 4. EMA (均线趋势过滤)
            ema200 = ta.trend.EMAIndicator(df['c'], window=200).ema_indicator().iloc[-1]
            # 价格在200日线上方，趋势看多，基础分加成
            trend_score = 80 if current_price > ema200 else 20
            
            # 5. ATR (用于计算止损，不参与打分，但需要返回)
            atr = ta.volatility.AverageTrueRange(df['h'], df['l'], df['c'], window=14).average_true_range().iloc[-1]
            
            # 综合技术分
            final_tech_score = (rsi_score*0.2 + macd_score*0.3 + kdj_score*0.2 + trend_score*0.3)
            
            return df, final_tech_score, atr, ema200
        except Exception as e:
            st.error(f"技术面分析出错: {e}")
            return None, 50, 0, 0

    def get_etf_score(self):
        """资金面分析"""
        try:
            ticker = yf.Ticker(self.etf)
            df = ticker.history(period="5d")
            if df.empty: return 50
            # 量价逻辑
            change = (df['Close'].iloc[-1] - df['Close'].iloc[-2])
            vol_ratio = df['Volume'].iloc[-1] / df['Volume'].mean()
            score = 50 + (20 * vol_ratio if change > 0 else -20 * vol_ratio)
            return max(0, min(100, score))
        except: return 50

    def get_whale_score(self, threshold):
        """链上巨鲸分析"""
        try:
            trades = self.exchange.fetch_trades(self.symbol, limit=500)
            df = pd.DataFrame(trades)
            df['cost'] = df['price'] * df['amount']
            whales = df[df['cost'] >= threshold]
            if whales.empty: return 50
            buy_vol = whales[whales['side'] == 'buy']['cost'].sum()
            total_vol = whales['cost'].sum()
            return (buy_vol / total_vol) * 100 if total_vol > 0 else 50
        except: return 50

    def get_news_score(self):
        """舆情分析"""
        try:
            keyword = 'Bitcoin' if 'BTC' in self.symbol else self.symbol.split('/')[0]
            rss = f"https://news.google.com/rss/search?q={keyword}+crypto+when:1d&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss)
            scores = [TextBlob(e.title).sentiment.polarity for e in feed.entries[:10]]
            if not scores: return 50
            avg = sum(scores) / len(scores)
            return (avg + 1) * 50 # 映射 -1~1 到 0~100
        except: return 50

# --- 4. 执行逻辑 ---

bot = QuantEngine(crypto_symbol, etf_ticker, timeframe, limit)

# 获取数据
with st.spinner('AI 正在全网扫描数据... (技术面 + 资金面 + 链上 + 舆情)'):
    df_k, s_tech, atr_val, ema_val = bot.get_tech_analysis()
    s_etf = bot.get_etf_score()
    s_whale = bot.get_whale_score(whale_threshold)
    s_news = bot.get_news_score()

# 计算总分
final_score = (s_tech * w_tech) + (s_etf * w_fund) + (s_whale * w_onchain) + (s_news * w_news)
current_price = df_k['c'].iloc[-1]

# --- 5. 界面展示 ---

# === 顶部：最终结论 ===
st.subheader("🤖 AI 决策报告")

col_gauge, col_plan = st.columns([1.5, 2])

with col_gauge:
    # 仪表盘绘制
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = final_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "多空胜率评分", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "white"},
            'steps': [
                {'range': [0, 40], 'color': '#FF4B4B'},   # 空
                {'range': [40, 60], 'color': '#808080'},  # 震荡
                {'range': [60, 100], 'color': '#00CC96'}  # 多
            ],
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20,r=20,t=30,b=20), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    
    # 状态解读
    status = ""
    color = ""
    if final_score >= 80: status, color = "🚀 极度看多 (Strong Buy)", "green"
    elif final_score >= 60: status, color = "🟢 谨慎做多 (Buy)", "green"
    elif final_score <= 20: status, color = "📉 极度看空 (Strong Sell)", "red"
    elif final_score <= 40: status, color = "🔴 谨慎做空 (Sell)", "red"
    else: status, color = "⚖️ 震荡观望 (Wait)", "gray"
    
    st.markdown(f"<h3 style='text-align: center; color: {color};'>{status}</h3>", unsafe_allow_html=True)

with col_plan:
    st.markdown("### 🎯 智能交易计划 (ATR动态风控)")
    st.info("基于 ATR (平均真实波幅) 计算的科学止损止盈位，拒绝凭感觉交易。")
    
    # 根据分数决定做多还是做空建议
    signal_side = "LONG (做多)" if final_score >= 50 else "SHORT (做空)"
    
    if final_score >= 50:
        # 做多计划
        stop_loss = current_price - (atr_val * 2) # 2倍ATR止损
        take_profit = current_price + (atr_val * 3) # 3倍ATR止盈
        entry_color = "green"
    else:
        # 做空计划 (假设合约交易)
        stop_loss = current_price + (atr_val * 2)
        take_profit = current_price - (atr_val * 3)
        entry_color = "red"
        
    p1, p2, p3 = st.columns(3)
    p1.metric("1. 建议入场价", f"${current_price:.2f}")
    p2.metric("2. 止损价 (SL)", f"${stop_loss:.2f}", delta=f"-{(atr_val*2):.2f}", delta_color="inverse")
    p3.metric("3. 止盈价 (TP)", f"${take_profit:.2f}", delta=f"+{(atr_val*3):.2f}")
    
    st.caption(f"💡 策略逻辑: {signal_side} | 盈亏比 1.5 : 1 | 当前波动率(ATR): {atr_val:.2f}")
    
    # 仓位建议
    confidence = abs(final_score - 50) * 2 # 0-100的信心度
    pos_size = "0%"
    if confidence > 80: pos_size = "20% (重仓)"
    elif confidence > 40: pos_size = "10% (标配)"
    else: pos_size = "0% (空仓观望)"
    
    st.markdown(f"**💰 建议仓位:** `{pos_size}` (信心度: {confidence:.0f}%)")

st.markdown("---")

# === 底部：因子拆解 (Explainable AI) ===
st.subheader("📊 为什么 AI 这么判断？ (因子归因)")

with st.expander("🔎 点击查看详细得分拆解", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.progress(int(s_tech))
        st.metric("技术面 (40%)", f"{s_tech:.0f} 分", "MACD+KDJ+EMA")
    with c2:
        st.progress(int(s_etf))
        st.metric("资金面 (30%)", f"{s_etf:.0f} 分", f"{etf_ticker} 流向")
    with c3:
        st.progress(int(s_whale))
        st.metric("链上 (20%)", f"{s_whale:.0f} 分", "巨鲸多空比")
    with c4:
        st.progress(int(s_news))
        st.metric("消息面 (10%)", f"{s_news:.0f} 分", "AI 舆情分析")

# K线图表
st.subheader("📈 趋势确认 (EMA + 信号)")

# 增加一个安全检查，防止 df_k 为空时报错
if df_k is not None and not df_k.empty:
    fig_k = go.Figure()
    
    # 1. 绘制 K 线
    fig_k.add_trace(go.Candlestick(
        x=df_k['ts'], 
        open=df_k['o'], 
        high=df_k['h'], 
        low=df_k['l'], 
        close=df_k['c'], 
        name='K线'
    ))
    
    # 2. 绘制 EMA 趋势线 (确保长度一致)
    # 创建一个与 K 线等长的 EMA 列表 (因为 ema_val 是一个单数值)
    ema_line = [ema_val] * len(df_k)
    
    fig_k.add_trace(go.Scatter(
        x=df_k['ts'], 
        y=ema_line, 
        line=dict(color='orange', width=2), 
        name='EMA200牛熊线'
    ))
    
    fig_k.update_layout(
        height=400, 
        template="plotly_dark", 
        xaxis_rangeslider_visible=False, 
        title=f"{crypto_symbol} vs EMA200趋势线"
    )
    
    # 注意：这里变量名必须是 fig_k，不能是 fig_
    st.plotly_chart(fig_k, use_container_width=True)

else:
    st.warning("⚠️ 暂无 K 线数据，可能是网络问题或 API 连接失败，请稍后刷新重试。")