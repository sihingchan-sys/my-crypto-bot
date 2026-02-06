import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
import feedparser
from textblob import TextBlob
from datetime import datetime, timedelta

# --- 1. 页面配置 ---
st.set_page_config(page_title="AI 量化指挥官 (终极全能版)", layout="wide", page_icon="🛡️")
st.title("🛡️ Crypto AI 指挥官 (Day 6 Ultimate Final)")

# --- 2. 侧边栏 (新增：实时行情看板) ---
with st.sidebar:
    st.header("🎛️ 指挥台")
    
    # === 1. 实时行情模块 (NEW) ===
    # 这里的逻辑独立于核心策略，只为了显示价格
    symbol = st.text_input("交易对", value='BTC-USD').upper()
    
    try:
        # 快速获取最新的一根 1分钟 K线
        live_df = yf.download(symbol, period='1d', interval='1m', progress=False)
        if not live_df.empty:
            if isinstance(live_df.columns, pd.MultiIndex): live_df.columns = live_df.columns.get_level_values(0)
            
            # 获取最新价和开盘价计算涨跌
            current_p = live_df['Close'].iloc[-1]
            open_p = live_df['Open'].iloc[0] # 当日开盘价
            high_p = live_df['High'].max()
            low_p = live_df['Low'].min()
            change = (current_p - open_p) / open_p * 100
            
            # 显示漂亮的指标卡
            st.markdown("### 🪙 实时行情")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric("最新价", f"${current_p:,.2f}", f"{change:.2f}%")
            with col_p2:
                st.caption(f"⬆️ 高: ${high_p:,.0f}")
                st.caption(f"⬇️ 低: ${low_p:,.0f}")
            
            # 刷新按钮 (Streamlit需要手动刷新才能更新价格)
            if st.button("🔄 刷新最新价", use_container_width=True):
                st.rerun()
        else:
            st.error("无法获取行情")
    except:
        st.warning("行情连接中...")

    st.divider()

    # === 2. 策略控制 (保持不变) ===
    tf_options = {'15m (短线)': '15m', '1h (波段)': '1h', '1d (长线)': '1d'} 
    tf_label = st.selectbox("作战周期", list(tf_options.keys()), index=1)
    tf = tf_options[tf_label]
    
    # 优化参数
    st.subheader("🧪 策略优化")
    use_ema_filter = st.checkbox("✅ 开启 EMA200 趋势过滤", value=True, help="勾选后，只做顺大势的单子（价格在均线上方只做多，下方只做空）。")
    backtest_days = st.slider("回测天数", 30, 365, 90)
    
    # 权重配置
    with st.expander("⚙️ AI 权重微调"):
        w_tech = st.slider("技术面", 0.0, 1.0, 0.4)
        w_fund = st.slider("资金面", 0.0, 1.0, 0.3)
        w_main = st.slider("主力面", 0.0, 1.0, 0.2)
        w_news = st.slider("消息面", 0.0, 1.0, 0.1)

    st.markdown("---")
    if st.button('🚀 启动全系统分析', type="primary"):
        st.rerun()
# --- 3. 核心全能引擎 (OptimizedCommander) ---

class OptimizedCommander:
    def __init__(self, symbol, tf):
        self.symbol = symbol
        self.tf = tf

    # === A. 数据获取 (含 EMA 预计算) ===
    def get_data(self):
        try:
            # 必须拿足够长的数据来算 EMA200
            period_map = {'15m': '20d', '1h': '6mo', '1d': '2y'}
            period = period_map.get(self.tf, '1mo')
            
            df = yf.download(self.symbol, period=period, interval=self.tf, progress=False)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = df.rename(columns={'Open': 'o', 'High': 'h', 'Low': 'l', 'Close': 'c', 'Volume': 'v'})
            df['ts'] = df.index
            
            # 计算 EMA200 用于趋势判断
            df['ema200'] = ta.trend.EMAIndicator(df['c'], window=200).ema_indicator()
            return df
        except: return None

    # === B. 核心策略 (含趋势过滤) ===
    def calculate_strategy(self, current_price, ref_df, current_ema=None, use_filter=False):
        if len(ref_df) < 2: return None
        last = ref_df.iloc[-2]
        
        # 1. Pivot 计算
        H, L, C = last['High'], last['Low'], last['Close']
        P = (H + L + C) / 3
        R1 = 2*P - L
        S1 = 2*P - H
        R2 = P + (H - L)
        S2 = P - (H - L)
        
        # 2. 原始方向判断
        raw_direction = "LONG" if current_price > P else "SHORT"
        
        # 3. 趋势过滤逻辑
        is_allowed = True
        filter_msg = ""
        
        if use_filter and current_ema is not None and not pd.isna(current_ema):
            if raw_direction == "LONG" and current_price < current_ema:
                is_allowed = False
                filter_msg = "(逆势被拦截 🚫)"
            elif raw_direction == "SHORT" and current_price > current_ema:
                is_allowed = False
                filter_msg = "(逆势被拦截 🚫)"
        
        # 4. 生成信号 (挂单逻辑)
        if raw_direction == "LONG":
            direction = f"LONG (做多) {filter_msg}"
            if current_price < R1: entry, tp, sl = P, R1, S1
            elif current_price < R2: entry, tp, sl = R1, R2, P
            else: entry, tp, sl = R2, R2*1.05, R1
        else:
            direction = f"SHORT (做空) {filter_msg}"
            if current_price > S1: entry, tp, sl = P, S1, R1
            elif current_price > S2: entry, tp, sl = S1, S2, P
            else: entry, tp, sl = S2, S2*0.95, S1
            
        return {
            'P': P, 'R1': R1, 'R2': R2, 'S1': S1, 'S2': S2,
            'dir': direction, 'entry': entry, 'tp': tp, 'sl': sl,
            'is_allowed': is_allowed, 
            'ref_date': last.name
        }

    # === C. 回测引擎 (带过滤) ===
    def run_backtest(self, days=90, use_filter=False):
        try:
            tf_map = {
                '15m': {'interval': '1d',  'period': f"{days+60}d", 'desc': '日线 (Daily)'},
                '1h':  {'interval': '1wk', 'period': '5y',          'desc': '周线 (Weekly)'},
                '1d':  {'interval': '1mo', 'period': '10y',         'desc': '月线 (Monthly)'}
            }
            cfg = tf_map.get(self.tf, tf_map['15m'])
            
            df = yf.download(self.symbol, period=cfg['period'], interval=cfg['interval'], progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if len(df) < 5: return None, 0, 0
            
            # 必须在回测周期上也算EMA，用于历史过滤
            df['ema_trend'] = ta.trend.EMAIndicator(df['Close'], window=20 if cfg['interval']!='1d' else 5).ema_indicator()
            
            df = df.iloc[:-1] 
            history = []
            wins = 0
            losses = 0
            
            for i in range(25, len(df)): 
                yesterday = df.iloc[i-1]
                today = df.iloc[i]
                
                ema_val = yesterday['ema_trend']
                strat = self.calculate_strategy(today['Open'], df.iloc[:i], ema_val, use_filter)
                
                if not strat or not strat['is_allowed']: continue 
                
                entry = strat['entry']
                tp = strat['tp']
                sl = strat['sl']
                is_long = "LONG" in strat['dir']
                
                did_enter = (today['Low'] <= entry <= today['High'])
                result = None
                pnl = 0
                
                if did_enter:
                    if is_long:
                        if today['Low'] <= sl:
                            result = "止损"
                            pnl = -1 * abs(entry - sl)
                            losses += 1
                        elif today['High'] >= tp:
                            result = "止盈"
                            pnl = abs(tp - entry)
                            wins += 1
                    else:
                        if today['High'] >= sl:
                            result = "止损"
                            pnl = -1 * abs(sl - entry)
                            losses += 1
                        elif today['Low'] <= tp:
                            result = "止盈"
                            pnl = abs(entry - tp)
                            wins += 1

                    if result:
                        history.append({
                            '日期': today.name.strftime('%Y-%m-%d'),
                            '过滤': "✅开启" if use_filter else "❌关闭",
                            '方向': "多" if is_long else "空",
                            '结果': result,
                            '盈亏': round(pnl, 2)
                        })
            
            res_df = pd.DataFrame(history)
            if not res_df.empty and cfg['interval'] == '1d': res_df = res_df.tail(days)
            return res_df, wins, losses
            
        except Exception as e:
            return None, 0, 0

    # === D. 辅助分析 (四维打分 + 详细数据) ===
    def analyze_score(self, df, etf_ticker, symbol):
        try:
            # 1. Tech
            if df is None: return 50, 50, 50, 50, 0
            rsi = ta.momentum.RSIIndicator(df['c']).rsi().iloc[-1]
            ema = df['ema200'].iloc[-1] if 'ema200' in df else df['c'].mean()
            s_tech = ( (50+(50-rsi)) + (80 if df['c'].iloc[-1]>ema else 20) ) / 2
            
            # 2. Fund
            s_fund = 50
            try:
                edf = yf.Ticker(etf_ticker).history(period="1mo")
                if not edf.empty:
                    chg = edf['Close'].iloc[-1] - edf['Close'].iloc[-2]
                    s_fund = 60 if chg > 0 else 40
            except: pass
                
            # 3. Main
            cmf = ta.volume.ChaikinMoneyFlowIndicator(df['h'], df['l'], df['c'], df['v'], window=20).chaikin_money_flow().iloc[-1]
            s_main = 50 + cmf*200
            
            # 4. News
            s_news = 50
            news_items = []
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

# --- 4. 执行逻辑 ---
bot = OptimizedCommander(symbol, tf)

with st.spinner('🚀 系统正在全速运转...'):
    # 1. 实时数据
    df_k = bot.get_data()
    curr_price = df_k['c'].iloc[-1] if df_k is not None else 0
    curr_ema = df_k['ema200'].iloc[-1] if df_k is not None else None
    
    # 2. 参考数据
    ref_config = {
        '15m': {'interval': '1d', 'period': '60d', 'name': '昨日日线'},
        '1h':  {'interval': '1wk', 'period': '2y', 'name': '上周周线'},
        '1d':  {'interval': '1mo', 'period': '5y', 'name': '上月月线'}
    }
    cfg = ref_config.get(tf, ref_config['15m'])
    ref_df = yf.download(symbol, period=cfg['period'], interval=cfg['interval'], progress=False)
    if isinstance(ref_df.columns, pd.MultiIndex): ref_df.columns = ref_df.columns.get_level_values(0)
    
    # 3. 策略计算
    plan = bot.calculate_strategy(curr_price, ref_df, curr_ema, use_ema_filter)
    
    # 4. 综合打分
    s_t, s_f, s_m, s_n, ema_val, news_list = bot.analyze_score(df_k, 'IBIT', symbol)
    final_score = s_t*w_tech + s_f*w_fund + s_m*w_main + s_n*w_news
    
    # 5. 跑回测
    backtest_df, wins, losses = bot.run_backtest(backtest_days, use_ema_filter)

# --- 5. 界面展示 (6 Tabs 完整版) ---

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 决策总览", "📈 技术分析", "🇺🇸 机构资金", "🐋 主力动向", "🗞️ 消息舆情", "🧪 优化回测"
])
# === Tab 1: 决策总览 (增强说明版) ===
with tab1:
    if plan:
        data_date = plan['ref_date'].strftime('%Y-%m-%d')
        st.caption(f"📅 策略基准: {data_date} | 周期: {cfg['name']} | 过滤: {'✅开启' if use_ema_filter else '❌关闭'}")
    
    c1, c2 = st.columns([1, 2])
    
    # --- 左侧：仪表盘 ---
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
        st.info(f"💡 **AI 信心度:** {confidence:.0f}%")

        # 🟢 新增：仪表盘说明 (折叠式)
        with st.expander("📖 如何看懂仪表盘?"):
            st.markdown("""
            **1. 指针颜色与方向:**
            - 🟩 **绿色区域 (60-100)**: 多头强势，建议**做多**。
            - 🟥 **红色区域 (0-40)**: 空头强势，建议**做空**。
            - ⚪ **白色区域 (40-60)**: 震荡市，建议**观望**。

            **2. 信心度 (仓位参考):**
            - **< 20%**: 信心不足，**空仓**或**轻仓**。
            - **> 50%**: 信心较强，可**正常仓位**。
            - **> 80%**: 极度确信，可适当**重仓**。
            """)

    # --- 右侧：指令卡 ---
    with c2:
        st.subheader(f"🎯 作战指令卡")
        if plan:
            if not plan['is_allowed']:
                st.warning(f"🚫 **信号被拦截：** 虽有 Pivot 信号，但当前价格逆势 (EMA200)，系统建议观望。")
            else:
                k1, k2, k3 = st.columns(3)
                k1.metric("建议挂单 (Entry)", f"${plan['entry']:.2f}", plan['dir'], help="在此价格挂【限价单 Limit Order】等待成交，不要市价追单。")
                k2.metric("止盈目标 (TP)", f"${plan['tp']:.2f}", help="建议在此价格分批止盈，落袋为安。")
                k3.metric("止损防守 (SL)", f"${plan['sl']:.2f}", delta_color="inverse", help="如果价格触及此线，必须无条件止损离场，保住本金。")
                
                st.success("✅ **信号有效：** 顺势交易，胜率较高。请注意这是挂单(Limit)策略。")

                # 🟢 新增：操作指南 (折叠式)
                with st.expander("🛠️ 实战操作指南 (新手必读)", expanded=True):
                    st.markdown(f"""
                    1. **挂单操作**: 打开交易所，选择 **{symbol}** 合约。
                    2. **下单类型**: 选择 **限价委托 (Limit)**。
                    3. **价格设置**: 
                       - 价格填上面的 **${plan['entry']:.2f}**。
                       - 止盈填 **${plan['tp']:.2f}**。
                       - 止损填 **${plan['sl']:.2f}**。
                    4. **有效期**: 
                       - 15m 周期: **每日早8点** 前未成交则撤单。
                       - 1h/1d 周期: 持有直到成交或趋势改变。
                    """)
        else:
            st.error("数据不足")

    st.markdown("---")
    if plan:
        st.subheader("🗺️ 战场地图 (挂单参考)")
        table_data = [
            {"代号": "R2", "角色": "🏔️ 天花板", "价格": plan['R2'], "说明": "极强阻力，到了可以止盈跑路"},
            {"代号": "R1", "角色": "🧱 阻力墙", "价格": plan['R1'], "说明": "普通阻力，可能遇阻回调"},
            {"代号": "P", "角色": "⚖️ 中轴线", "价格": plan['P'],  "说明": "多空分界，上方看多，下方看空"},
            {"代号": "S1", "角色": "🛡️ 地板", "价格": plan['S1'], "说明": "第一支撑，跌到这可以尝试接多"},
            {"代号": "S2", "角色": "🌋 岩浆", "价格": plan['S2'], "说明": "最后防线，跌破则趋势反转"},
        ]
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

# === Tab 2: 技术分析 ===
with tab2:
    if df_k is not None:
        fig_k = go.Figure(go.Candlestick(x=df_k['ts'], open=df_k['o'], high=df_k['h'], low=df_k['l'], close=df_k['c']))
        fig_k.add_trace(go.Scatter(x=df_k['ts'], y=df_k['ema200'], line=dict(color='orange', width=2), name='EMA200'))
        if plan and plan['is_allowed']:
            fig_k.add_hline(y=plan['entry'], line_dash="dash", line_color="blue", annotation_text="Entry")
            fig_k.add_hline(y=plan['tp'], line_dash="dot", line_color="green", annotation_text="TP")
            fig_k.add_hline(y=plan['sl'], line_dash="dot", line_color="red", annotation_text="SL")
        fig_k.update_layout(height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_k, use_container_width=True)

# === Tab 3-5: 其他板块 ===
with tab3: 
    st.metric("资金面得分", f"{s_f:.0f}")
    st.info("数据来源: IBIT ETF 资金流向")
with tab4: 
    st.metric("主力面得分 (CMF)", f"{s_m:.0f}")
    st.caption("高于50主力吸筹，低于50主力出货")
with tab5: 
    st.metric("新闻情绪分", f"{s_n:.0f}")
    for n in news_list:
        st.markdown(f"- {n.title}")

# === Tab 6: 优化回测 ===
with tab6:
    st.subheader(f"📊 回测报告 (过滤: {'开' if use_ema_filter else '关'})")
    if backtest_df is not None and not backtest_df.empty:
        total = wins + losses
        rate = (wins/total*100) if total else 0
        pnl = backtest_df['盈亏'].sum()
        
        m1, m2, m3 = st.columns(3)
        m1.metric("交易次数", f"{total}")
        m2.metric("胜率", f"{rate:.1f}%")
        m3.metric("总盈亏", f"${pnl:.2f}")
        
        st.dataframe(backtest_df, use_container_width=True)
        backtest_df['累计盈亏'] = backtest_df['盈亏'].cumsum()
        st.line_chart(backtest_df['累计盈亏'])
    else:
        st.info("⚠️ 该时间段内无符合条件的交易 (可能是被趋势过滤拦截了)。")
