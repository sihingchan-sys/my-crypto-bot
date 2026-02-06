import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
import feedparser
from textblob import TextBlob
from datetime import datetime, timedelta
import os

# --- 1. 页面配置 ---
st.set_page_config(page_title="AI 量化指挥官 (钻石防重版)", layout="wide", page_icon="🛡️")
st.title("🛡️ Crypto AI 指挥官 (Day 6 Diamond Fix)")

# --- 2. 核心全能引擎 (逻辑层) ---

class OptimizedCommander:
    def __init__(self, symbol, tf):
        self.symbol = symbol
        self.tf = tf
        self.history_file = 'ai_signal_history_v3.csv' # 升级文件名，强制使用新格式

    # === A. 数据获取 ===
    def get_data(self):
        try:
            period_map = {'15m': '20d', '1h': '6mo', '1d': '2y'}
            period = period_map.get(self.tf, '1mo')
            df = yf.download(self.symbol, period=period, interval=self.tf, progress=False)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = df.rename(columns={'Open': 'o', 'High': 'h', 'Low': 'l', 'Close': 'c', 'Volume': 'v'})
            df['ts'] = df.index
            df['ema200'] = ta.trend.EMAIndicator(df['c'], window=200).ema_indicator()
            return df
        except: return None

    # === B. 核心策略 ===
    def calculate_strategy(self, current_price, ref_df, current_ema=None, use_filter=False):
        if len(ref_df) < 2: return None
        last = ref_df.iloc[-2]
        
        H, L, C = last['High'], last['Low'], last['Close']
        P = (H + L + C) / 3
        R1, S1 = 2*P - L, 2*P - H
        R2, S2 = P + (H - L), P - (H - L)
        
        raw_direction = "LONG" if current_price > P else "SHORT"
        is_allowed, filter_msg = True, ""
        
        if use_filter and current_ema is not None and not pd.isna(current_ema):
            if raw_direction == "LONG" and current_price < current_ema: is_allowed, filter_msg = False, "(逆势拦截)"
            elif raw_direction == "SHORT" and current_price > current_ema: is_allowed, filter_msg = False, "(逆势拦截)"
        
        if raw_direction == "LONG":
            direction = f"做多 {filter_msg}"
            if current_price < R1: entry, tp, sl = P, R1, S1
            elif current_price < R2: entry, tp, sl = R1, R2, P
            else: entry, tp, sl = R2, R2*1.05, R1
        else:
            direction = f"做空 {filter_msg}"
            if current_price > S1: entry, tp, sl = P, S1, R1
            elif current_price > S2: entry, tp, sl = S1, S2, P
            else: entry, tp, sl = S2, S2*0.95, S1
            
        return {
            'P': P, 'R1': R1, 'R2': R2, 'S1': S1, 'S2': S2,
            'dir': direction, 'entry': entry, 'tp': tp, 'sl': sl,
            'is_allowed': is_allowed, 'ref_date': last.name, 'raw_dir': raw_direction
        }

    # === C. 回测引擎 ===
    def run_backtest(self, days=90, use_filter=False):
        try:
            tf_map = {'15m': {'interval': '1d', 'period': f"{days+60}d"}, '1h': {'interval': '1wk', 'period': '5y'}, '1d': {'interval': '1mo', 'period': '10y'}}
            cfg = tf_map.get(self.tf, tf_map['15m'])
            
            df = yf.download(self.symbol, period=cfg['period'], interval=cfg['interval'], progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if len(df) < 5: return None, 0, 0
            
            window = 20 if cfg['interval'] != '1d' else 5
            df['ema_trend'] = ta.trend.EMAIndicator(df['Close'], window=window).ema_indicator()
            df = df.iloc[:-1] 
            
            history, wins, losses = [], 0, 0
            start_idx = max(25, window + 5)
            if len(df) <= start_idx: return None, 0, 0

            for i in range(start_idx, len(df)): 
                yesterday, today = df.iloc[i-1], df.iloc[i]
                strat = self.calculate_strategy(today['Open'], df.iloc[:i], yesterday['ema_trend'], use_filter)
                if not strat or not strat['is_allowed']: continue 
                
                entry, tp, sl = strat['entry'], strat['tp'], strat['sl']
                is_long = "做多" in strat['dir']
                
                if (today['Low'] <= entry <= today['High']):
                    res, pnl = None, 0
                    if is_long:
                        if today['Low'] <= sl: res, pnl = "止损", -1 * abs(entry-sl); losses += 1
                        elif today['High'] >= tp: res, pnl = "止盈", abs(tp-entry); wins += 1
                    else:
                        if today['High'] >= sl: res, pnl = "止损", -1 * abs(sl-entry); losses += 1
                        elif today['Low'] <= tp: res, pnl = "止盈", abs(entry-tp); wins += 1
                    
                    if res: history.append({'日期': today.name.strftime('%Y-%m-%d'), '方向': "多" if is_long else "空", '结果': res, '盈亏': round(pnl, 2)})
            
            res_df = pd.DataFrame(history)
            if not res_df.empty and cfg['interval'] == '1d': res_df = res_df.tail(days)
            return res_df, wins, losses
        except: return None, 0, 0

    # === D. 自动记录与审计 ===
    def audit_history(self):
        if not os.path.exists(self.history_file): return pd.DataFrame()
        
        df = pd.read_csv(self.history_file)
        if df.empty: return df
        
        # 审计数据获取
        audit_data = yf.download(self.symbol, period='60d', interval='1d', progress=False)
        if audit_data.empty: return df
        if isinstance(audit_data.columns, pd.MultiIndex): audit_data.columns = audit_data.columns.get_level_values(0)
        
        updated = False
        
        for index, row in df.iterrows():
            if "⏳" in str(row['结果']):
                try:
                    signal_date = pd.to_datetime(row['记录时间']).date()
                    future_data = audit_data[audit_data.index.date >= signal_date]
                    
                    entry = float(row['挂单价'])
                    tp = float(row['止盈'])
                    sl = float(row['止损'])
                    is_long = row['方向'] == "多"
                    
                    for idx, day in future_data.iterrows():
                        if day['Low'] <= entry <= day['High']:
                            status = None
                            close_price = 0
                            if is_long:
                                if day['Low'] <= sl: status = "❌止损"; close_price = sl; updated=True
                                elif day['High'] >= tp: status = "🏆止盈"; close_price = tp; updated=True
                            else:
                                if day['High'] >= sl: status = "❌止损"; close_price = sl; updated=True
                                elif day['Low'] <= tp: status = "🏆止盈"; close_price = tp; updated=True
                            
                            if status:
                                df.at[index, '结果'] = status
                                df.at[index, '平仓价'] = close_price
                                break
                except: pass
            
        if updated:
            df.to_csv(self.history_file, index=False)
            
        return df

    # 🔥🔥🔥 核弹级防重逻辑 🔥🔥🔥
    def save_signal(self, plan, score):
        if not plan: return
        if not plan['is_allowed']: return

        # 核心：使用【基准日期】作为防伪ID
        # 15m信号的ref_date是昨天，只要昨天没变，信号就不该变
        ref_date_str = plan['ref_date'].strftime('%Y-%m-%d')
        current_entry = round(plan['entry'], 2)
        current_dir = "多" if "做多" in plan['dir'] else "空"
        
        new_record = {
            '记录时间': datetime.now().strftime('%Y-%m-%d %H:%M'),
            '交易对': self.symbol,
            '周期': self.tf,
            '基准日期': ref_date_str, # 新增列：防重核心
            '方向': current_dir,
            '挂单价': current_entry,
            '平仓价': 0,
            '止盈': round(plan['tp'], 2),
            '止损': round(plan['sl'], 2),
            'AI信心': int(score),
            '结果': '⏳挂单中'
        }
        
        # 1. 如果文件不存在，直接保存
        if not os.path.exists(self.history_file):
            pd.DataFrame([new_record]).to_csv(self.history_file, index=False)
            return

        # 2. 读取现有数据
        df = pd.read_csv(self.history_file)
        
        # 3. 超级严格的检查
        # 规则：如果在历史记录里，找到了 [同交易对] + [同周期] + [同基准日期] + [同方向] + [同价格] 的记录
        # 那么，绝对禁止保存！不管你是几点刷新的。
        if not df.empty:
            # 兼容旧文件没有 '基准日期' 的情况 (虽然建议删文件，但防止万一)
            if '基准日期' not in df.columns:
                df['基准日期'] = '0000-00-00' # 填充默认值
            
            # 强制转字符串比对，消灭浮点误差
            # 检查：有没有一条记录，它的基准日期 == 今天的基准日期 AND 挂单价 == 今天的挂单价
            duplicate_check = df[
                (df['交易对'] == self.symbol) &
                (df['周期'] == self.tf) &
                (df['基准日期'] == ref_date_str) & 
                (df['方向'] == current_dir) &
                (df['挂单价'].astype(str) == str(current_entry)) 
            ]
            
            if not duplicate_check.empty:
                # print("发现重复信号，拦截保存！") # 调试用
                return 

        # 4. 通过检查，保存
        pd.DataFrame([new_record]).to_csv(self.history_file, mode='a', header=False, index=False)

    # === E. 辅助分析 ===
    def analyze_score(self, df, etf_ticker, symbol):
        try:
            if df is None: return 50, 50, 50, 50, 0, []
            rsi = ta.momentum.RSIIndicator(df['c']).rsi().iloc[-1]
            ema = df['ema200'].iloc[-1] if 'ema200' in df else df['c'].mean()
            s_tech = ( (50+(50-rsi)) + (80 if df['c'].iloc[-1]>ema else 20) ) / 2
            
            s_fund = 50
            try:
                edf = yf.Ticker(etf_ticker).history(period="1mo")
                if not edf.empty:
                    chg = edf['Close'].iloc[-1] - edf['Close'].iloc[-2]
                    s_fund = 60 if chg > 0 else 40
            except: pass
                
            cmf = ta.volume.ChaikinMoneyFlowIndicator(df['h'], df['l'], df['c'], df['v'], window=20).chaikin_money_flow().iloc[-1]
            s_main = 50 + cmf*200
            
            s_news, news_items = 50, []
            try:
                kw = 'Bitcoin' if 'BTC' in symbol else symbol.split('-')[0]
                rss = f"https://news.google.com/rss/search?q={kw}+crypto+when:1d&hl=en-US&gl=US&ceid=US:en"
                feed = feedparser.parse(rss)
                scores = [TextBlob(e.title).sentiment.polarity for e in feed.entries[:5]]
                if scores: s_news = (sum(scores)/len(scores) + 1) * 50
                news_items = feed.entries[:5]
            except: pass
            
            return s_tech, s_fund, s_main, s_news, ema, news_items
        except: return 50, 50, 50, 50, 0, []

# --- 3. 执行逻辑 ---
st.sidebar.header("🎛️ 指挥台")

# A. 实时行情
symbol = st.sidebar.text_input("交易对", value='BTC-USD').upper()
try:
    live_df = yf.download(symbol, period='1d', interval='1m', progress=False)
    if not live_df.empty:
        if isinstance(live_df.columns, pd.MultiIndex): live_df.columns = live_df.columns.get_level_values(0)
        curr_p, open_p = live_df['Close'].iloc[-1], live_df['Open'].iloc[0]
        change_p = (curr_p - open_p) / open_p * 100
        st.sidebar.markdown(f"**最新**: ${curr_p:,.2f}")
        st.sidebar.markdown(f"**涨跌**: :{'red' if change_p < 0 else 'green'}[{change_p:.2f}%]")
        if st.sidebar.button("🔄 刷新数据"): st.rerun()
except: pass

st.sidebar.divider()

# B. 策略控制
tf_options = {'15m (短线)': '15m', '1h (波段)': '1h', '1d (长线)': '1d'} 
tf = tf_options[st.sidebar.selectbox("作战周期", list(tf_options.keys()), index=1)]
use_ema_filter = st.sidebar.checkbox("✅ 开启 EMA 过滤", value=True)
backtest_days = st.sidebar.slider("回测天数", 30, 365, 90)

# 权重微调
with st.sidebar.expander("⚙️ 权重设置"):
    w_tech = st.slider("技术", 0.0, 1.0, 0.4)
    w_fund = st.slider("资金", 0.0, 1.0, 0.3)
    w_main = st.slider("主力", 0.0, 1.0, 0.2)
    w_news = st.slider("舆情", 0.0, 1.0, 0.1)

# 初始化
bot = OptimizedCommander(symbol, tf)

with st.spinner('🚀 正在全速运转...'):
    df_k = bot.get_data()
    curr_price = df_k['c'].iloc[-1] if df_k is not None else 0
    curr_ema = df_k['ema200'].iloc[-1] if df_k is not None else None
    
    ref_config = {'15m': '1d', '1h': '1wk', '1d': '1mo'}
    ref_df = yf.download(symbol, period='2y', interval=ref_config.get(tf, '1d'), progress=False)
    if isinstance(ref_df.columns, pd.MultiIndex): ref_df.columns = ref_df.columns.get_level_values(0)
    
    plan = bot.calculate_strategy(curr_price, ref_df, curr_ema, use_ema_filter)
    s_t, s_f, s_m, s_n, ema_val, news_list = bot.analyze_score(df_k, 'IBIT', symbol)
    final_score = s_t*w_tech + s_f*w_fund + s_m*w_main + s_n*w_news
    
    bot.save_signal(plan, final_score)
    hist_df = bot.audit_history()
    backtest_df, wins, losses = bot.run_backtest(backtest_days, use_ema_filter)

# === 侧边栏：实盘战绩 (分频道增强版) ===
st.sidebar.divider()
st.sidebar.subheader("🏆 实盘战绩 (审计)")

def render_stats(df_target, title_prefix):
    if df_target.empty:
        st.sidebar.caption(f"暂无 {title_prefix} 记录")
        return
    
    # 统计
    real_wins = len(df_target[df_target['结果'].str.contains("止盈")])
    real_losses = len(df_target[df_target['结果'].str.contains("止损")])
    total_real = real_wins + real_losses
    real_rate = (real_wins / total_real * 100) if total_real > 0 else 0
    
    c1, c2 = st.sidebar.columns(2)
    c1.metric(f"{title_prefix}完结", f"{total_real}单")
    c2.metric("真实胜率", f"{real_rate:.0f}%", delta="实战")
    
    st.sidebar.caption(f"📜 {title_prefix} 记录 (最新5条):")
    display_cols = ['记录时间','方向','挂单价','平仓价','结果']
    # 兼容旧数据防止报错
    valid_cols = [c for c in display_cols if c in df_target.columns]
    hist_display = df_target[valid_cols].tail(5).iloc[::-1].copy()
    if '平仓价' in hist_display.columns:
        hist_display['平仓价'] = hist_display['平仓价'].apply(lambda x: f"{x:.2f}" if float(x) > 0 else "-")
    st.sidebar.dataframe(hist_display, hide_index=True)

if not hist_df.empty:
    t_all, t_15m, t_1h, t_1d = st.sidebar.tabs(["全部", "15m", "1h", "1d"])
    with t_all: render_stats(hist_df, "全部")
    with t_15m: render_stats(hist_df[hist_df['周期'] == '15m'], "15m")
    with t_1h: render_stats(hist_df[hist_df['周期'] == '1h'], "1h")
    with t_1d: render_stats(hist_df[hist_df['周期'] == '1d'], "1d")
else:
    st.sidebar.info("暂无实盘记录，等待信号...")

# === 主界面 Tabs ===
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🏠 决策", "📈 技术", "🇺🇸 资金", "🐋 主力", "🗞️ 舆情", "🧪 回测"])

with tab1:
    c1, c2 = st.columns([1, 2])
    with c1:
        fig_g = go.Figure(go.Indicator(mode="gauge+number", value=final_score, gauge={'axis': {'range': [0, 100]}, 'steps': [{'range': [0, 40], 'color': '#FF4B4B'}, {'range': [60, 100], 'color': '#00CC96'}]}))
        fig_g.update_layout(height=250, margin=dict(t=30,b=20,l=20,r=20))
        st.plotly_chart(fig_g, use_container_width=True)
        confidence = abs(final_score - 50) * 2
        st.info(f"💡 AI 信心度: {confidence:.0f}%")
        
        with st.expander("📖 如何看懂仪表盘 (仓位参考)?", expanded=False):
            st.markdown("""
            **1. 指针区域与方向:**
            - 🟩 **绿色 (60-100)**: 多头强势 -> **只做多**
            - 🟥 **红色 (0-40)**: 空头强势 -> **只做空**
            - ⚪ **白色 (40-60)**: 震荡不明 -> **观望**

            **2. 信心度与仓位管理:**
            - **< 20%**: 看不懂 -> **空仓休息** 😴
            - **20% - 50%**: 有把握 -> **轻仓试水** (10%本金) 💧
            - **> 60%**: 极度确信 -> **正常/重仓** (30%+ 本金) 💰
            """)

    with c2:
        if plan and plan['is_allowed']:
            k1, k2, k3 = st.columns(3)
            k1.metric("挂单 Entry", f"${plan['entry']:.2f}", plan['dir'])
            k2.metric("止盈 TP", f"${plan['tp']:.2f}")
            k3.metric("止损 SL", f"${plan['sl']:.2f}", delta_color="inverse")
            st.success("✅ 信号有效：请在交易所挂限价单 (Limit Order)。")
            with st.expander("🛠️ 实战操作指南 (新手必读)", expanded=True):
                st.markdown(f"1. **{symbol}** 开 **限价单(Limit)**。\n2. 价格 **{plan['entry']:.2f}** | 止盈 **{plan['tp']:.2f}** | 止损 **{plan['sl']:.2f}**。\n3. **{tf}** 周期，未成交请勿追单。")
        else:
            st.warning("🚫 信号被拦截：当前逆势或数据不足，建议观望。")

    st.markdown("---")
    if plan:
        st.subheader("🗺️ 战场地图")
        table_data = [
            {"代号": "R2", "价格": plan['R2'], "说明": "天花板/强阻力"},
            {"代号": "R1", "价格": plan['R1'], "说明": "阻力墙/止盈点"},
            {"代号": "P",  "价格": plan['P'],  "说明": "中轴/多空分界"},
            {"代号": "S1", "价格": plan['S1'], "说明": "地板/接多点"},
            {"代号": "S2", "价格": plan['S2'], "说明": "岩浆/强支撑"},
        ]
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

with tab2:
    if df_k is not None:
        fig_k = go.Figure(go.Candlestick(x=df_k['ts'], open=df_k['o'], high=df_k['h'], low=df_k['l'], close=df_k['c']))
        fig_k.add_trace(go.Scatter(x=df_k['ts'], y=df_k['ema200'], line=dict(color='orange'), name='EMA200'))
        if plan and plan['is_allowed']:
            fig_k.add_hline(y=plan['entry'], line_dash="dash", line_color="blue", annotation_text="Entry")
            fig_k.add_hline(y=plan['tp'], line_dash="dot", line_color="green", annotation_text="TP")
            fig_k.add_hline(y=plan['sl'], line_dash="dot", line_color="red", annotation_text="SL")
        fig_k.update_layout(height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_k, use_container_width=True)
        with st.expander("📚 指标说明"):
            st.caption("🍊 EMA200: 牛熊分界线。🔵 Pivot: 挂单系统。")

with tab3:
    st.metric("资金面评分", f"{s_f:.0f}", delta="基于ETF流向")
    st.subheader("🏛️ ETF 资金流向")
    cols = st.columns(4)
    for i, t in enumerate(['IBIT', 'FBTC', 'BITB', 'ARKB']):
        try:
            d = yf.Ticker(t).history(period="5d")
            if not d.empty: cols[i].metric(t, f"${d['Close'].iloc[-1]:.2f}", f"{(d['Close'].iloc[-1]-d['Close'].iloc[-2])/d['Close'].iloc[-2]*100:.2f}%")
        except: pass

with tab4:
    st.metric("CMF 主力吸筹分", f"{s_m:.0f}", delta=">50吸筹" if s_m>50 else "出货")
    if df_k is not None:
        nv = ((df_k['c'] - df_k['o']) / (df_k['h'] - df_k['l'])) * df_k['v']
        st.plotly_chart(go.Figure(go.Bar(x=df_k['ts'], y=nv, marker_color=['#00CC96' if v>0 else '#FF4B4B' for v in nv])).update_layout(height=250, title="资金净流向"), use_container_width=True)

with tab5:
    st.metric("AI 舆情情绪分", f"{s_n:.0f}", delta=">50乐观")
    st.subheader("🗞️ 舆情简报")
    for n in news_list: st.markdown(f"- [{n.title}]({n.link})")

with tab6:
    if backtest_df is not None and not backtest_df.empty:
        tot = wins+losses
        st.metric("回测胜率 (非实盘)", f"{(wins/tot*100) if tot else 0:.1f}%", f"总盈亏 ${backtest_df['盈亏'].sum():.2f}")
        st.dataframe(backtest_df, use_container_width=True)
    else: st.info("无回测记录")
