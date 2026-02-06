import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
import feedparser
from textblob import TextBlob
from datetime import datetime

# --- 1. 页面配置 ---
st.set_page_config(page_title="AI 量化指挥官 (融合修正版)", layout="wide", page_icon="🛸")
st.title("🛸 Crypto AI 指挥官 (Day 6 Ultimate Plus)")

# --- 2. 侧边栏 ---
with st.sidebar:
    st.header("🎛️ 指挥台")
    symbol = st.text_input("交易对 (Yahoo格式)", value='BTC-USD').upper()
    
    tf_options = {
        '15m (短线突击)': '15m', 
        '1h (波段战役)': '1h', 
        '1d (趋势远征)': '1d'
    } 
    tf_label = st.selectbox("作战周期", list(tf_options.keys()), index=1)
    tf = tf_options[tf_label]
    
    etf_ticker = st.text_input("美股 ETF", value='IBIT')
    
    st.divider()
    
    with st.expander("⚙️ 权重微调"):
        w_tech = st.slider("技术面", 0.0, 1.0, 0.4)
        w_fund = st.slider("资金面", 0.0, 1.0, 0.3)
        w_main = st.slider("主力面", 0.0, 1.0, 0.2)
        w_news = st.slider("消息面", 0.0, 1.0, 0.1)

    if st.button('🚀 执行最终策略', type="primary"):
        st.rerun()

# --- 3. 核心全能引擎 ---

class GrandCommander:
    def __init__(self, symbol, tf):
        self.symbol = symbol
        self.tf = tf

    def get_data(self):
        try:
            download_period = "5d" if self.tf == "15m" else "2y"
            df = yf.download(self.symbol, period=download_period, interval=self.tf, progress=False)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = df.rename(columns={'Open': 'o', 'High': 'h', 'Low': 'l', 'Close': 'c', 'Volume': 'v'})
            df['ts'] = df.index
            return df
        except: return None

    def get_pivots_and_plan(self, current_price):
        try:
            # 自适应获取参考数据
            ref_config = {
                '15m': {'interval': '1d', 'period': '1mo', 'name': '昨日日线'},
                '1h':  {'interval': '1wk', 'period': '3mo', 'name': '上周周线'},
                '1d':  {'interval': '1mo', 'period': '2y',  'name': '上月月线'}
            }
            cfg = ref_config.get(self.tf, ref_config['15m'])
            
            ref_df = yf.download(self.symbol, period=cfg['period'], interval=cfg['interval'], progress=False)
            if isinstance(ref_df.columns, pd.MultiIndex): ref_df.columns = ref_df.columns.get_level_values(0)
            
            if len(ref_df) < 2: return None
            last = ref_df.iloc[-2] 

            # 计算 Fibonacci Pivot Points
            H, L, C = last['High'], last['Low'], last['Close']
            P = (H + L + C) / 3
            R1 = 2*P - L
            S1 = 2*P - H
            R2 = P + (H - L)
            S2 = P - (H - L)
            R3 = H + 2 * (P - L)
            S3 = L - 2 * (H - P)
            
            # 生成策略 (基于位置)
            if current_price > P:
                direction = "LONG"
                if current_price < R1: entry, tp, sl = P, R1, S1
                elif current_price < R2: entry, tp, sl = R1, R2, P
                else: entry, tp, sl = R2, R3, R1
            else:
                direction = "SHORT"
                if current_price > S1: entry, tp, sl = P, S1, R1
                elif current_price > S2: entry, tp, sl = S1, S2, P
                else: entry, tp, sl = S2, S3, S1

            return {
                'P': P, 'R1': R1, 'R2': R2, 'S1': S1, 'S2': S2, 'R3': R3, 'S3': S3,
                'dir': direction, 'entry': entry, 'tp': tp, 'sl': sl,
                'ref_name': cfg['name']
            }
        except: return None

    # --- 四维分析模块 (保持不变) ---
    def analyze_tech(self, df):
        if df is None: return 50, 0
        rsi = ta.momentum.RSIIndicator(df['c']).rsi().iloc[-1]
        ema200 = ta.trend.EMAIndicator(df['c'], window=200).ema_indicator().iloc[-1]
        if pd.isna(ema200): ema200 = df['c'].mean()
        # 趋势得分：价格在均线上方得80，否则20
        trend_s = 80 if df['c'].iloc[-1] > ema200 else 20
        # RSI得分：超卖(30)得高分，超买(70)得低分
        rsi_s = 50 + (50 - rsi)
        return (rsi_s + trend_s)/2, ema200

    def analyze_fund(self, ticker):
        try:
            df = yf.Ticker(ticker).history(period="1mo")
            if df.empty: return 50, None
            change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
            vol = df['Volume'].iloc[-1] / df['Volume'].mean()
            # 资金流入(涨+放量)加分，流出减分
            score = 50 + (20 * vol if change > 0 else -20 * vol)
            return max(0, min(100, score)), df
        except: return 50, None

    def analyze_main(self, df):
        if df is None: return 50
        # CMF指标判断主力意图
        cmf = ta.volume.ChaikinMoneyFlowIndicator(df['h'], df['l'], df['c'], df['v'], window=20).chaikin_money_flow().iloc[-1]
        return max(0, min(100, 50 + cmf*200))

    def analyze_news(self, symbol):
        try:
            kw = 'Bitcoin' if 'BTC' in symbol else symbol.split('-')[0]
            rss = f"https://news.google.com/rss/search?q={kw}+crypto+when:1d&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss)
            scores = [TextBlob(e.title).sentiment.polarity for e in feed.entries[:10]]
            if not scores: return 50, []
            return (sum(scores)/len(scores) + 1) * 50, feed.entries[:5]
        except: return 50, []

# --- 4. 执行运算 ---
bot = GrandCommander(symbol, tf)

with st.spinner('AI 正在进行多维共振分析...'):
    df_k = bot.get_data()
    
    if df_k is not None:
        curr_price = df_k['c'].iloc[-1]
        
        # 1. 制定基础计划 (数学层)
        plan = bot.get_pivots_and_plan(curr_price)
        
        # 2. 计算 AI 得分 (智能层)
        s_tech, ema_val = bot.analyze_tech(df_k)
        s_fund, df_etf = bot.analyze_fund(etf_ticker)
        s_main = bot.analyze_main(df_k)
        s_news, news_list = bot.analyze_news(symbol)
        
        final_score = s_tech*w_tech + s_fund*w_fund + s_main*w_main + s_news*w_news
    else:
        st.error("无法连接全球数据节点，请稍后重试。")
        st.stop()

# --- 5. 界面展示 ---

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 决策总览", "📈 技术分析", "🇺🇸 机构资金", "🐋 主力动向", "🗞️ 消息舆情"
])

# === Tab 1: 决策总览 (核心修改区) ===
with tab1:
    c1, c2 = st.columns([1, 2])
    
    # A. 仪表盘
    with c1:
        st.subheader("AI 胜率仪表盘")
        fig_g = go.Figure(go.Indicator(
            mode = "gauge+number", value = final_score,
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "white"},
                     'steps': [{'range': [0, 40], 'color': '#FF4B4B'}, {'range': [60, 100], 'color': '#00CC96'}]}
        ))
        fig_g.update_layout(height=250, margin=dict(t=30,b=20,l=20,r=20))
        st.plotly_chart(fig_g, use_container_width=True)
        confidence = abs(final_score - 50) * 2
        st.info(f"AI 综合得分: **{final_score:.1f}**\n\n信心度: {confidence:.0f}%")

    # B. 智能指令卡 (加入共振逻辑)
    with c2:
        ref_title = f" (基于 {plan['ref_name']})" if plan else ""
        st.subheader(f"🎯 AI 共振指令卡{ref_title}")
        
        if plan:
            # === 🔥 核心逻辑修正：检查 AI 分数与 Pivot 方向是否冲突 ===
            
            is_conflict = False
            conflict_msg = ""
            
            # 1. 冲突检测
            if plan['dir'] == "LONG" and final_score < 40:
                is_conflict = True
                conflict_msg = "⛔ **指令驳回：** Pivot 结构看涨，但 AI 综合评分过低 (<40，看空)。\n\n**建议：** 多头动能不足，放弃做多，等待观望。"
                
            elif plan['dir'] == "SHORT" and final_score > 60:
                is_conflict = True
                conflict_msg = "⛔ **指令驳回：** Pivot 结构看空，但 AI 综合评分过高 (>60，看多)。\n\n**建议：** 空头风险较大，放弃做空，等待观望。"
            
            # 2. 展示结果
            if is_conflict:
                st.warning(conflict_msg)
                # 即使冲突，也可以显示个灰色的参考价，但弱化它
                st.caption(f"*(仅供参考：结构化支撑位在 ${plan['entry']:.2f})*")
            
            else:
                # 共振成功！显示绿色/红色通行证
                color_str = "green" if plan['dir']=="LONG" else "red"
                direction_cn = "做多 (Long)" if plan['dir']=="LONG" else "做空 (Short)"
                
                k1, k2, k3 = st.columns(3)
                k1.metric("1. 挂单开仓价", f"${plan['entry']:.2f}", direction_cn)
                k2.metric("2. 目标止盈 (TP)", f"${plan['tp']:.2f}", delta=f"预期 {(plan['tp']-plan['entry'])/plan['entry']:.2%}")
                k3.metric("3. 宽幅止损 (SL)", f"${plan['sl']:.2f}", delta=f"风险 {(plan['sl']-plan['entry'])/plan['entry']:.2%}", delta_color="inverse")
                
                # 等待提示
                wait_dist = abs(curr_price - plan['entry']) / curr_price
                if wait_dist > 0.005:
                    st.info(f"⏳ **耐心等待：** 请在 {plan['entry']:.2f} 挂单，不要追单。")
                else:
                    st.success(f"⚡ **立刻执行：** 现价已到达最佳开仓区！")

        else:
            st.error("数据不足，无法生成指令")

    st.markdown("---")
    
    # C. 地图与解释
    st.subheader("🗺️ 战场地图")
    if plan:
        # 显示地图... (保持原样)
        table_data = [
            {"代号": "R2", "价格": f"${plan['R2']:.2f}", "说明": "强阻力/目标位"},
            {"代号": "R1", "价格": f"${plan['R1']:.2f}", "说明": "弱阻力/第一止盈"},
            {"代号": "P",  "价格": f"${plan['P']:.2f}",  "说明": "多空分界线"},
            {"代号": "S1", "价格": f"${plan['S1']:.2f}", "说明": "弱支撑/第一止盈"},
            {"代号": "S2", "价格": f"${plan['S2']:.2f}", "说明": "强支撑/防守位"},
        ]
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

# === Tab 2-5 (保持原样，无需改动) ===
# ... (为节省篇幅，这里复用上面的 Tab 2-5 代码即可)
with tab2:
    st.subheader(f"📈 {symbol} 趋势全景")
    if df_k is not None:
        display_limit = 200
        df_view = df_k.tail(display_limit)
        fig_k = go.Figure()
        fig_k.add_trace(go.Candlestick(x=df_view['ts'], open=df_view['o'], high=df_view['h'], low=df_view['l'], close=df_view['c'], name='K线'))
        ema_plot = [ema_val] * len(df_view) 
        fig_k.add_trace(go.Scatter(x=df_view['ts'], y=ema_plot, line=dict(color='orange', width=2), name='EMA200'))
        if plan:
            fig_k.add_hline(y=plan['P'], line_dash="dash", line_color="yellow", annotation_text="P")
            fig_k.add_hline(y=plan['R1'], line_dash="dot", line_color="red", annotation_text="R1")
            fig_k.add_hline(y=plan['S1'], line_dash="dot", line_color="green", annotation_text="S1")
        fig_k.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_k, use_container_width=True)

with tab3:
    st.subheader("🇺🇸 机构资金面")
    if df_etf is not None:
        st.metric("资金面得分", f"{s_fund:.0f}")
        st.line_chart(df_etf['Close'])
    else: st.info("ETF 数据暂缺")

with tab4:
    st.subheader("🐋 主力吸筹/派发")
    st.metric("主力得分 (CMF)", f"{s_main:.0f}")
    st.line_chart(df_k['v']) # 简单展示成交量

with tab5:
    st.subheader("🗞️ 消息舆情")
    st.metric("AI 情绪分", f"{s_news:.0f}")
    for n in news_list:
        st.markdown(f"- {n.title}")
