import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
import feedparser
from textblob import TextBlob
from datetime import datetime, timedelta
import os
import ccxt
import requests

# === 插入到 class OptimizedCommander 之前 ===
class TradeLogger:
    def __init__(self, filename='my_trade_journal.csv'):
        self.filename = filename
        
    def load_log(self):
        if os.path.exists(self.filename):
            return pd.read_csv(self.filename)
        else:
            return pd.DataFrame(columns=['记录时间', '交易对', '周期', '方向', '投入金额(U)', '开仓价', '平仓价', '状态', '盈亏(U)', '收益率(%)'])

    def add_trade(self, symbol, tf, direction, entry, amount):
        df = self.load_log()
        new_row = {
            '记录时间': datetime.now().strftime('%Y-%m-%d %H:%M'),
            '交易对': symbol,
            '周期': tf,
            '方向': direction,
            '投入金额(U)': float(amount),
            '开仓价': float(entry),
            '平仓价': 0.0,
            '状态': '⏳挂单中', 
            '盈亏(U)': 0.0,
            '收益率(%)': 0.0
        }
        df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)
        df.to_csv(self.filename, index=False)

    def save_log(self, df):
        for i, row in df.iterrows():
            if row['状态'] in ['✅止盈', '❌止损', '🚀交易中'] and float(row['平仓价']) > 0:
                entry = float(row['开仓价'])
                close = float(row['平仓价'])
                amt = float(row['投入金额(U)'])
                # 计算盈亏
                pnl = (close - entry) / entry * amt if '多' in row['方向'] else (entry - close) / entry * amt
                roi = (close - entry) / entry * 100 if '多' in row['方向'] else (entry - close) / entry * 100
                df.at[i, '盈亏(U)'] = round(pnl, 2)
                df.at[i, '收益率(%)'] = round(roi, 2)
            elif row['状态'] == '🗑️撤单':
                df.at[i, '盈亏(U)'] = 0
                df.at[i, '收益率(%)'] = 0
        df.to_csv(self.filename, index=False)

# --- 1. 页面配置 ---
st.set_page_config(page_title="AI 量化指挥官 (US IP 修复版)", layout="wide", page_icon="🛡️")
st.title("🛡️ Crypto AI 指挥官 (Day 6 Final Fix)")

# --- 2. 核心全能引擎 (逻辑层) ---

class OptimizedCommander:
    def __init__(self, symbol, tf):
        self.symbol = symbol
        self.tf = tf 

    # === A. 数据获取 (修正版：修复 ts 丢失问题) ===
    def get_data(self):
        try:
            import ccxt
            # 1. 初始化交易所
            try:
                exchange = ccxt.kraken({'timeout': 3000})
                symbol_map = {'BTC-USD': 'BTC/USD', 'ETH-USD': 'ETH/USD'}
                target_symbol = symbol_map.get(self.symbol, self.symbol.replace('-', '/'))
                # 抓取数据
                timeframe_map = {'15m': '15m', '1h': '60m', '1d': '1440m'} # Kraken有时需要特定格式，通用尝试直接传
                ohlcv = exchange.fetch_ohlcv(target_symbol, self.tf, limit=300)
            except:
                # 备用 Gate
                exchange = ccxt.gate({'timeout': 3000})
                target_symbol = self.symbol.replace('-', '_')
                ohlcv = exchange.fetch_ohlcv(target_symbol, self.tf, limit=300)

            # 2. 整理数据
            df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            df['ts'] = pd.to_datetime(df['ts'], unit='ms') 
            df.set_index('ts', inplace=True) # ts 变成了索引
            
            # 🔥🔥🔥 关键修复在这里：把索引复制回列 🔥🔥🔥
            df['ts'] = df.index 
            
            # 3. 计算指标
            df['ema200'] = ta.trend.EMAIndicator(df['c'], window=200).ema_indicator()
            
            return df
            
        except Exception as e:
            # print(f"CCXT 失败，回退到 Yahoo: {e}")
            # 降级方案：使用 yfinance
            try:
                period_map = {'15m': '5d', '1h': '1mo', '1d': '1y'}
                period = period_map.get(self.tf, '1mo')
                df = yf.download(self.symbol, period=period, interval=self.tf, progress=False)
                if df.empty: return None
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df = df.rename(columns={'Open': 'o', 'High': 'h', 'Low': 'l', 'Close': 'c', 'Volume': 'v'})
                
                # Yahoo 也是同样的处理逻辑
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
        
        # === H. AI 参数自适应引擎 (Pro版：ATR动态风控 + 趋势感知) ===
    def ai_optimize_parameters(self, days=30):
        """
        AI 进化方向：
        1. 不再使用固定百分比止损，而是使用 ATR (波动率) 倍数。
        2. 引入 ADX 过滤：趋势弱时不硬做趋势单。
        3. 评分标准：不再只看利润，引入胜率权重 (利润 * 胜率)，防止“一次暴富、九次爆仓”的参数胜出。
        """
        try:
            # 1. 获取数据
            df = self.get_data()
            if df is None or len(df) < 200: return {'sl_multiplier': 2.0, 'rr': 1.5, 'mode': 'Unknown'}
            
            # 计算切片
            rows_per_day = 24 if '1h' in self.tf else (96 if '15m' in self.tf else 1)
            train_len = days * rows_per_day
            train_df = df.iloc[-train_len:].copy() if len(df) > train_len else df.copy()
            
            # --- 🤖 智能指标计算 ---
            # A. 计算 ATR (波动率尺子)
            train_df['atr'] = ta.volatility.AverageTrueRange(train_df['h'], train_df['l'], train_df['c'], window=14).average_true_range()
            # B. 计算 ADX (趋势强度尺子)
            train_df['adx'] = ta.trend.ADXIndicator(train_df['h'], train_df['l'], train_df['c'], window=14).adx()
            
            # 2. 定义搜索空间 (更高级的参数)
            # 止损不再是 %，而是 ATR 的倍数 (1.5倍波动, 2倍波动...)
            atr_mult_range = [1.5, 2.0, 2.5, 3.0] 
            rr_range = [1.0, 1.5, 2.0, 3.0]
            
            best_score = -9999
            best_params = {'sl_multiplier': 2.0, 'rr': 1.5, 'mode': '震荡(默认)'}
            
            # 3. 智能回测循环
            for atr_mult in atr_mult_range:
                for rr in rr_range:
                    total_pnl = 0
                    wins = 0
                    total_trades = 0
                    
                    # 模拟交易逻辑
                    ema_col = train_df['ema200']
                    close_col = train_df['c']
                    atr_col = train_df['atr']
                    adx_col = train_df['adx']
                    
                    for i in range(1, len(train_df)-1):
                        # 过滤：如果是趋势策略，要求 ADX > 20 才开单 (避免在死鱼盘里频繁止损)
                        if adx_col.iloc[i] < 20: continue 
                        
                        price = close_col.iloc[i]
                        atr = atr_col.iloc[i]
                        
                        # 简单的趋势跟随信号
                        if price > ema_col.iloc[i]: 
                            entry = price
                            # 🔥 智能止损：当前价格减去 N 倍的波动率
                            stop_loss_dist = atr * atr_mult 
                            sl = entry - stop_loss_dist
                            tp = entry + (stop_loss_dist * rr)
                            
                            # 往后看
                            future = train_df.iloc[i+1:min(i+20, len(train_df))]
                            if future.empty: continue
                            
                            if future['l'].min() <= sl:
                                total_pnl -= 1 # 亏损 1R
                                total_trades += 1
                            elif future['h'].max() >= tp:
                                total_pnl += rr # 盈利 RR
                                wins += 1
                                total_trades += 1
                                
                    # 4. 📝 智能评分系统 (Sharpe Ratio 简化版)
                    # 我们不只看总利润，还要看胜率。
                    # 得分 = 总利润 * (胜率权重)
                    if total_trades > 0:
                        win_rate = wins / total_trades
                        # 惩罚低胜率：如果胜率低于 40%，分数打折
                        penalty = 1.0 if win_rate > 0.4 else 0.5
                        
                        final_score = total_pnl * penalty
                        
                        if final_score > best_score:
                            best_score = final_score
                            # 判断当前环境
                            current_adx = adx_col.iloc[-1]
                            market_mode = "🔥单边趋势" if current_adx > 25 else "🌊震荡整理"
                            
                            best_params = {
                                'sl_multiplier': atr_mult, 
                                'rr': rr,
                                'mode': market_mode
                            }
                            
            return best_params
            
        except Exception as e:
            # print(f"智能训练出错: {e}")
            return {'sl_multiplier': 2.0, 'rr': 1.5, 'mode': '错误'}
        # === C. 获取资金费率 (增强版：优先币安 -> 备用Gate -> 兜底默认) ===
    def get_funding_rate(self):
        try:
            # 方案 1: 优先尝试币安 (Binance) - 最权威
            # 注意：如果网络不通，这里会迅速超时跳到方案 2
            url = "https://fapi.binance.com/fapi/v1/premiumIndex"
            symbol_str = self.symbol.split('-')[0] + "USDT" # 格式转换: BTC-USD -> BTCUSDT
            params = {'symbol': symbol_str}
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            r = requests.get(url, params=params, headers=headers, timeout=3)
            if r.status_code == 200:
                data = r.json()
                rate = float(data['lastFundingRate'])
                return rate # 成功！直接返回 (例如 0.0001)
        except:
            pass # 币安失败，默默进入下一步

        try:
            # 方案 2: 强力备用 Gate.io (无需翻墙，CCXT直连)
            # Gate 的 API 在国内通常比币安好连
            import ccxt
            exchange = ccxt.gate({
                'enableRateLimit': True, 
                'timeout': 3000, 
                'options': {'defaultType': 'swap'} # 指定请求合约数据
            })
            
            # 格式转换: BTC-USD -> BTC_USDT
            target_symbol = self.symbol.replace('-', '_') 
            
            funding = exchange.fetch_funding_rate(target_symbol)
            rate = float(funding['fundingRate'])
            
            # 🛡️ 数据清洗：防止出现 -25% 这种乌龙
            # 正常费率通常在 -0.01 到 0.01 之间。如果绝对值 > 0.5 (50%)，肯定是数据源错了
            if abs(rate) > 0.5: 
                return 0.0001 # 数据异常，返回默认值
            
            return rate
        except Exception as e:
            # print(f"Gate获取失败: {e}") # 调试用，平时可以注释掉
            pass
            
        # 方案 3: 最后的倔强 (兜底值)
        # 如果所有交易所都连不上，为了不让程序报错崩溃，返回标准牛市费率
        return 0.0001 # 对应 0.01%            
    # === G. 综合打分 ===
    def analyze_score(self, df, etf_ticker, symbol):
        # 初始化默认值
        s_tech, s_fund, s_main, s_news, ema, news_items = 50, 50, 50, 50, 0, []
        s_funding_score, funding_msg = 50, "获取失败"

        try:
            # 1. 技术面
            if df is not None:
                rsi = ta.momentum.RSIIndicator(df['c']).rsi().iloc[-1]
                ema = df['ema200'].iloc[-1] if 'ema200' in df else df['c'].mean()
                s_tech = ( (50+(50-rsi)) + (80 if df['c'].iloc[-1]>ema else 20) ) / 2
            
            # 2. 资金面 (ETF)
            try:
                edf = yf.Ticker(etf_ticker).history(period="1mo")
                if not edf.empty:
                    chg = edf['Close'].iloc[-1] - edf['Close'].iloc[-2]
                    s_fund = 60 if chg > 0 else 40
            except: pass
            
            # 3. 主力面 (CMF)
            if df is not None:
                cmf = ta.volume.ChaikinMoneyFlowIndicator(df['h'], df['l'], df['c'], df['v'], window=20).chaikin_money_flow().iloc[-1]
                s_main = 50 + cmf*200
            
            # 4. 舆情面
            try:
                kw = 'Bitcoin' if 'BTC' in symbol else symbol.split('-')[0]
                rss = f"https://news.google.com/rss/search?q={kw}+crypto+when:1d&hl=en-US&gl=US&ceid=US:en"
                feed = feedparser.parse(rss)
                scores = [TextBlob(e.title).sentiment.polarity for e in feed.entries[:5]]
                if scores: s_news = (sum(scores)/len(scores) + 1) * 50
                news_items = feed.entries[:5]
            except: pass

            # 5. 费率面 (NEW)
            # === 资金费率智能处理 (Auto-Fix) ===
            funding_rate = self.get_funding_rate()
            
            if funding_rate is not None:
                # 1. 默认尝试：假设它是标准小数 (如 0.0001)
                fr_percent = funding_rate * 100 
                
                # 2. 第一次纠错：如果结果太吓人 (超过 ±5%)
                # 说明原始数据可能已经是百分数了 (如 -0.25)
                if abs(fr_percent) > 5:
                    fr_percent = funding_rate # 直接用原始值 (-0.25%)
                
                # 3. 第二次纠错：如果还是太吓人 (超过 ±5%)
                # 说明原始数据可能是整数基点 (如 -25)
                if abs(fr_percent) > 5:
                    fr_percent = funding_rate / 100 # 除以100 (-0.25%)
                
                # 4. 格式化显示
                funding_msg = f"{fr_percent:.4f}%"
                
                # 5. 评分逻辑 (基于正确的百分比打分)
                # 费率 > 0.03% (万三) -> 危险，多头太挤
                if fr_percent > 0.03: s_funding_score = 20
                # 费率 > 0.01% (万一) -> 略热
                elif fr_percent > 0.01: s_funding_score = 40
                # 费率 < -0.01% (负费率) -> 机会，空头太挤
                elif fr_percent < -0.01: s_funding_score = 80
                # 费率 < 0 -> 偏多头利好
                elif fr_percent < 0: s_funding_score = 60
                else: s_funding_score = 50
            
            else:
                # 如果完全获取不到
                s_funding_score = 50
                funding_msg = "暂无数据"
                          
        except Exception as e: 
            print(f"分析出错: {e}")
            pass
        
        return s_tech, s_fund, s_main, s_news, ema, news_items, s_funding_score, funding_msg      

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

# 🔥🔥🔥 关键修复：必须在这里先初始化机器人！🔥🔥🔥
# 只有先定义了 bot，后面的 AI 和记账功能才能正常工作
bot = OptimizedCommander(symbol, tf) 

# === C. 我的实盘账本 (手工版) ===
logger = TradeLogger() # 初始化记账员

st.sidebar.divider()
st.sidebar.subheader("📓 我的实盘账本")
log_df = logger.load_log()

if not log_df.empty:
    # 算总账
    total_pnl = log_df['盈亏(U)'].sum()
    done_count = len(log_df[log_df['状态'].isin(['✅止盈', '❌止损'])])
    win_count = len(log_df[log_df['盈亏(U)'] > 0])
    win_rate = (win_count / done_count * 100) if done_count > 0 else 0
    
    c1, c2 = st.sidebar.columns(2)
    c1.metric("累计盈亏", f"${total_pnl:.2f}", delta_color="normal" if total_pnl>=0 else "inverse")
    c2.metric("实战胜率", f"{win_rate:.0f}%")

    st.sidebar.caption("👇 在下方直接修改状态和平仓价 (Enter保存):")
    # 可编辑表格
    edited_df = st.sidebar.data_editor(
        log_df,
        column_config={
            "状态": st.column_config.SelectboxColumn("状态", options=['⏳挂单中', '🚀交易中', '✅止盈', '❌止损', '🗑️撤单'], required=True),
            "平仓价": st.column_config.NumberColumn("平仓价", min_value=0, step=0.1, format="$%.2f"),
            "投入金额(U)": st.column_config.NumberColumn(format="$%.0f"),
            "盈亏(U)": st.column_config.NumberColumn(format="$%.2f", disabled=True),
        },
        hide_index=True,
        num_rows="dynamic"
    )
    # 保存修改
    if not edited_df.equals(log_df):
        logger.save_log(edited_df)
        st.rerun()
else:
    st.sidebar.info("暂无交易记录，快去决策页开单吧！")

# === D. AI 参数计算 (只计算，不显示，避免报错) ===
if st.sidebar.checkbox("🤖 开启 Pro级 AI 自适应", value=True):
    with st.sidebar.status("🧠 AI 正在计算 ATR 波动率与 ADX 趋势...", expanded=True) as status:
        # 因为 bot 已经在上面初始化了，所以这里不会再报错了！
        best_params = bot.ai_optimize_parameters(days=30)
        status.update(label="✅ 智能分析完成！", state="complete", expanded=False)
else:
    best_params = None        
  
    
# (原来的 if best_params: 以及后面的一大堆显示代码，统统删掉！)
with st.spinner('🚀 正在全速运转...'):
    df_k = bot.get_data()
    curr_price = df_k['c'].iloc[-1] if df_k is not None else 0
    curr_ema = df_k['ema200'].iloc[-1] if df_k is not None else None
    
    ref_config = {'15m': '1d', '1h': '1wk', '1d': '1mo'}
    ref_df = yf.download(symbol, period='2y', interval=ref_config.get(tf, '1d'), progress=False)
    if isinstance(ref_df.columns, pd.MultiIndex): ref_df.columns = ref_df.columns.get_level_values(0)
    
    plan = bot.calculate_strategy(curr_price, ref_df, curr_ema, use_ema_filter)
    # ... (上面是 plan = bot.calculate_strategy(...) )

    # === 🔥 AI 智能风控 (修正版：紧跟策略信号) ===
    # 只有当 1.策略有计划 2.AI算出了参数 时，才显示建议
    if plan and plan['is_allowed'] and best_params:
        
        # 1. 获取当前 ATR (用于计算宽窄)
        df_curr = bot.get_data()
        current_atr = ta.volatility.AverageTrueRange(df_curr['h'], df_curr['l'], df_curr['c']).average_true_range().iloc[-1]
        
        # 2. 读取主策略的信号 (关键修正！)
        strategy_entry = plan['entry']            # 你的开仓价 (Pivot点位)
        is_long = "做多" in plan['dir']            # 你的方向
        
        # 3. 利用 AI 参数计算 止盈/止损
        # 止损距离 = ATR * AI倍数
        sl_dist = current_atr * best_params['sl_multiplier']
        tp_dist = sl_dist * best_params['rr']
        
        if is_long:
            ai_sl = strategy_entry - sl_dist
            ai_tp = strategy_entry + tp_dist
            dir_icon = "🟢 做多 (Long)"
        else: # 做空
            ai_sl = strategy_entry + sl_dist
            ai_tp = strategy_entry - tp_dist
            dir_icon = "🔴 做空 (Short)"

        # 4. 显示在侧边栏 (虽然代码在这里，但可以用 st.sidebar 投射过去)
        st.sidebar.markdown("---")
        st.sidebar.success(f"🧠 **AI 优化建议 (基于当前信号)**")
        
        st.sidebar.info(f"""
        **针对开仓价 ${strategy_entry:.2f} 的 {dir_icon} 建议：**
        
        🛡️ **AI 止损**: **${ai_sl:.2f}**
        *(距离 -{sl_dist:.2f})*
        
        🎯 **AI 止盈**: **${ai_tp:.2f}**
        *(距离 +{tp_dist:.2f})*
        
        ---
        📊 **参数逻辑**: 
        止损 = {best_params['sl_multiplier']} x ATR
        盈亏比 = 1:{best_params['rr']} ({best_params.get('mode', '')})
        """)
    
    # 接收参数
    s_t, s_f, s_m, s_n, ema_val, news_list, s_fr, fr_msg = bot.analyze_score(df_k, 'IBIT', symbol)
    
    # 加权公式
    final_score = s_t*0.4 + s_f*0.2 + s_m*0.2 + s_fr*0.2
    
    backtest_df, wins, losses = bot.run_backtest(backtest_days, use_ema_filter)

# === 主界面 Tabs ===
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🏠 决策", "📈 技术", "🇺🇸 资金", "🐋 主力", "🗞️ 舆情", "🧪 回测"])

with tab1:
    c1, c2 = st.columns([1, 2])
    with c1:
        # 仪表盘
        fig_g = go.Figure(go.Indicator(mode="gauge+number", value=final_score, gauge={'axis': {'range': [0, 100]}, 'steps': [{'range': [0, 40], 'color': '#FF4B4B'}, {'range': [60, 100], 'color': '#00CC96'}]}))
        fig_g.update_layout(height=250, margin=dict(t=30,b=20,l=20,r=20))
        # 修复警告：Plotly 保持 use_container_width=True (这是新版推荐写法)
        st.plotly_chart(fig_g, use_container_width=True)
        
        # 🔥 找回丢失的 UI：信心度显示
        confidence = abs(final_score - 50) * 2
        st.info(f"💡 AI 信心度: {confidence:.0f}%")
        
        # 🔥 找回丢失的 UI：详细说明
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

   # === 替换 Tab 1 下的 with c2: 里面的内容 ===
    with c2:
        if plan and plan['is_allowed']:
            # 原有的显示指标代码
            k1, k2, k3 = st.columns(3)
            k1.metric("挂单 Entry", f"${plan['entry']:.2f}", plan['dir'])
            k2.metric("止盈 TP", f"${plan['tp']:.2f}")
            k3.metric("止损 SL", f"${plan['sl']:.2f}", delta_color="inverse")
            
            st.divider()
            st.markdown("### 📝 战术记录板")
            
            # --- 新增：开单表单 ---
            with st.form("manual_trade_form"):
                col_a, col_b = st.columns(2)
                with col_a:
                    # 默认投入 100U，你可以自己改默认值
                    trade_amt = st.number_input("本单投入 (USDT)", min_value=10.0, value=100.0, step=10.0)
                with col_b:
                    st.markdown("<br>", unsafe_allow_html=True)
                    # 提交按钮
                    submit = st.form_submit_button("⚡ 一键记录本单")
                
                if submit:
                    # 自动读取当前方向
                    raw_dir = "多" if "做多" in plan['dir'] else "空"
                    # 写入日志
                    logger.add_trade(symbol, tf, raw_dir, plan['entry'], trade_amt)
                    st.success(f"✅ 已记录：{symbol} {raw_dir} @ {plan['entry']:.2f}")
                    st.rerun() # 刷新立刻显示
            # ---------------------
            
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
        # 修复警告：DataFrame 移除 use_container_width
        st.dataframe(pd.DataFrame(table_data))

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
    st.subheader("🇺🇸 资金 & 📊 费率")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("🇺🇸 ETF 资金面", f"{s_f:.0f}分", help="美国现货ETF资金流向评分")
        if s_f > 50: st.caption("✅ 华尔街机构正在 **净买入**")
        else: st.caption("❌ 华尔街机构正在 **净流出**")
        
    with c2:
        st.metric("📊 合约资金费率", fr_msg, f"{s_fr}分", delta_color="normal" if s_fr==50 else "inverse", help="永续合约资金费率")
        # 智能解读文案
        if s_fr < 40: st.caption("⚠️ **费率过高 (+)**: 多头太拥挤，小心主力砸盘！")
        elif s_fr > 60: st.caption("🚀 **费率负值 (-)**: 空头太拥挤，可能暴力拉升！")
        else: st.caption("⚖️ **费率正常**: 多空力量均衡。")

    # === 🔥 新增：教科书级解释 (点击展开) ===
    with st.expander("📚 新手必读：如何看懂资金与费率？(点击展开)", expanded=False):
        st.markdown("""
        ### 1. 🇺🇸 ETF 资金面 (代表：贝莱德/富达)
        这是 **"聪明钱" (Smart Money)** 的动向，代表美国机构投资者的态度。
        * **📈 机构买入**: 说明华尔街看好后市，愿意真金白银接盘。 -> **长期利好 (底气足)**
        * **📉 机构卖出**: 说明机构在套现离场。 -> **长期利空 (抛压大)**

        ### 2. 📊 合约资金费率 (代表：市场情绪/拥挤度)
        这是 **"反向指标"**，用来判断市场是不是"疯了"。
        * **🔴 费率为正 (+)**: **多头付钱给空头**。说明做多的人非常多。
            * **0.01%**: 正常牛市情绪。
            * **> 0.03% (危险)**: 全网都在无脑做多，**车太重了**。主力往往会故意**暴跌**来清算这些多头 (多杀多)。
        * **🟢 费率为负 (-)**: **空头付钱给多头**。说明做空的人非常多。
            * **< 0%**: 市场情绪悲观。
            * **< -0.01% (机会)**: 全网都在无脑做空。主力往往会故意**暴涨**来打爆空头 (轧空/逼空)。
            
        **👉 口诀：费率太高不追多，费率太低不追空。**
        """)

    st.divider()
    
    st.subheader("🏛️ ETF 资金流向 (最近5天)")
    st.caption("观察 IBIT (贝莱德) 和 FBTC (富达) 的涨跌幅，它们是市场的风向标。")
    cols = st.columns(4)
    for i, t in enumerate(['IBIT', 'FBTC', 'BITB', 'ARKB']):
        try:
            d = yf.Ticker(t).history(period="5d")
            if not d.empty: 
                change = (d['Close'].iloc[-1]-d['Close'].iloc[-2])/d['Close'].iloc[-2]*100
                cols[i].metric(t, f"${d['Close'].iloc[-1]:.2f}", f"{change:.2f}%")
        except: pass

with tab4:
    st.subheader("🐋 主力 & 资金流")
    
    # 1. 核心指标显示
    # 使用 help 参数提供悬停提示
    st.metric("CMF 主力吸筹分", f"{s_m:.0f}分", delta="吸筹 (进场)" if s_m > 50 else "出货 (离场)", help="基于 Chaikin Money Flow (CMF) 计算的主力意图评分")
    
    # 智能解读文案
    if s_m > 60:
        st.caption("🟢 **强力吸筹**: 大户/机构正在**买入**，底部支撑较强。")
    elif s_m < 40:
        st.caption("🔴 **强力出货**: 大户/机构正在**抛售**，顶部压力巨大。")
    else:
        st.caption("⚪ **洗盘/观望**: 主力动作不明显，市场处于震荡期。")

    # 2. 资金流向图表 (可视化)
    if df_k is not None:
        # 计算每一根K线的资金净量 (Net Volume)
        # 逻辑：如果收盘价 > 开盘价，视为买入量；反之视为卖出量
        nv = ((df_k['c'] - df_k['o']) / (df_k['h'] - df_k['l'])) * df_k['v']
        
        fig_cmf = go.Figure(go.Bar(
            x=df_k['ts'], 
            y=nv, 
            marker_color=['#00CC96' if v>0 else '#FF4B4B' for v in nv],
            name="资金净量"
        ))
        fig_cmf.update_layout(
            height=300, 
            title="📊 资金净流向 (Net Volume Flow)",
            margin=dict(t=40, b=20, l=20, r=20),
            yaxis_title="成交量力度"
        )
        st.plotly_chart(fig_cmf, use_container_width=True)

    # 3. 教科书级解释 (Expander)
    st.divider()
    with st.expander("📚 新手必读：如何看懂主力吸筹 (CMF)？", expanded=False):
        st.markdown("""
        ### 🐋 什么是“主力” (Whales)？
        主力通常指拥有巨额资金的机构、交易所冷钱包或超级大户。他们的买卖行为往往决定了未来的趋势方向。

        ### 📊 评分逻辑 (基于 CMF 指标)
        AI 使用 **Chaikin Money Flow (CMF)** 来监控资金是 **流进** 还是 **流出**。
        
        * **🟢 吸筹 (Accumulation) [分数 > 50]**: 
            * **现象**: 收盘价经常收在最高价附近，且伴随大成交量。
            * **含义**: 主力在偷偷买入，把价格托住，通常是**拉升前兆**。
            
        * **🔴 出货 (Distribution) [分数 < 50]**: 
            * **现象**: 收盘价经常收在最低价附近，且伴随大成交量。
            * **含义**: 主力在趁反弹偷偷卖出，通常是**砸盘前兆**。

        ### 🔥 进阶战法：顶底背离
        * **底背离 (买入神技)**: 当 **价格在创新低**，但 **主力分却在变高**。
            * *解读*: 散户在恐慌割肉，但主力在悄悄抄底。 -> **强烈看涨**
        * **顶背离 (逃顶神技)**: 当 **价格在创新高**，但 **主力分却在变低**。
            * *解读*: 价格虽然在涨（诱多），但主力已经在撤退了。 -> **强烈看跌**
        """)

with tab5:
    st.metric("AI 舆情情绪分", f"{s_n:.0f}", delta=">50乐观")
    st.subheader("🗞️ 舆情简报")
    for n in news_list: st.markdown(f"- [{n.title}]({n.link})")

with tab6:
    if backtest_df is not None and not backtest_df.empty:
        tot = wins+losses
        st.metric("回测胜率 (非实盘)", f"{(wins/tot*100) if tot else 0:.1f}%", f"总盈亏 ${backtest_df['盈亏'].sum():.2f}")
        st.dataframe(backtest_df)
    else: st.info("无回测记录")
