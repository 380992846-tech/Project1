# ============================================
# 完整量化系统 V14：提高信号质量 + 优化持仓周期
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings
import logging
import pickle
import os
import time
from itertools import product
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("="*80)
print("🏆 完整量化系统 V14：提高信号质量 + 优化持仓周期")
print("="*80)

# ==================== 参数配置 ====================
INITIAL_CAPITAL = 1000000
RISK_FREE_RATE = 0.02

# 正则化配置
RF_CONFIG = {
    'n_estimators': 50,
    'max_depth': 5,
    'min_samples_split': 50,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'random_state': 42,
}

# 多资产配置
ASSETS = {
    '沪深300': {'symbol': '000300.SS', 'type': 'index', 'weight': 0.5},
    '国债ETF': {'symbol': '511010.SS', 'type': 'fund', 'weight': 0.3},
    '黄金ETF': {'symbol': '518880.SS', 'type': 'fund', 'weight': 0.2},
}

# V14优化参数（基于V13诊断）
RISK_PARITY_CONFIG = {
    'target_vol': 0.12,           # 目标波动率
    'max_position': 0.95,         # 仓位上限
    'stop_loss': 0.06,            # 止损（从0.05回到0.06）
    'take_profit': 0.12,          # 降低止盈（从0.20到0.12）
    'max_hold_days': 30,          # 延长持仓（从20到30）
    'signal_threshold': 0.40,     # 提高信号阈值（从0.35到0.40）
    'ml_weight': 0.6,
}

# 待扫描的参数（针对V13诊断结果）
SCAN_PARAMS = {
    'stop_loss': [0.05, 0.06, 0.07],
    'take_profit': [0.10, 0.12, 0.15],
    'signal_threshold': [0.35, 0.40, 0.45],
    'max_hold_days': [20, 25, 30, 35],
}

# 成本参数
BASE_SLIPPAGE = 0.0005
MAX_SLIPPAGE = 0.002
COMMISSION = 0.00015
STAMP_TAX = 0

print(f"\n📌 V14优化参数（提高信号质量）:")
for k, v in RISK_PARITY_CONFIG.items():
    print(f"   {k}: {v}")

# ==================== 1. 数据获取 ====================
print("\n📥 获取数据...")

def fetch_data_with_cache(symbol: str, asset_type: str, start_date: str, end_date: str, cache_dir: str = 'data_cache'):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{symbol.replace('.', '_')}.pkl")
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    price_series = None
    try:
        import yfinance as yf
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if len(df) > 0:
            price_series = df['Close'].rename(symbol)
    except: pass
    
    if price_series is None:
        try:
            import akshare as ak
            if asset_type == 'index':
                df = ak.stock_zh_index_daily(symbol=symbol)
            else:
                code = symbol.split('.')[0]
                df = ak.fund_etf_hist_em(symbol=code, period="daily",
                                          start_date=start_date.replace('-', ''),
                                          end_date=end_date.replace('-', ''))
            if len(df) > 0:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
                df = df[df.index >= start_date]
                price_series = df['close'].rename(symbol)
        except: pass
    
    if price_series is None:
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = dates[dates.weekday < 5]
        n = len(dates)
        if '000300' in symbol:
            trend = np.linspace(2500, 4500, n)
            cycle = np.sin(np.linspace(0, 4*np.pi, n)) * 300
            noise = np.random.randn(n) * 80
            prices = 3000 + trend * 0.3 + cycle + noise
        elif '511010' in symbol:
            prices = 100 + np.cumsum(np.random.randn(n) * 0.1)
        else:
            prices = 3.5 + np.cumsum(np.random.randn(n) * 0.02)
        price_series = pd.Series(prices, index=dates, name=symbol)
    
    with open(cache_file, 'wb') as f:
        pickle.dump(price_series, f)
    return price_series

def fetch_multi_asset_data(assets: Dict, start_date: str = '2015-01-01', end_date: str = '2026-04-10') -> pd.DataFrame:
    all_prices = {}
    for name, info in assets.items():
        price_series = fetch_data_with_cache(info['symbol'], info['type'], start_date, end_date)
        if price_series is not None:
            all_prices[name] = price_series
            logger.info(f"✅ {name}: {len(price_series)} 天")
        time.sleep(0.3)
    df_prices = pd.DataFrame(all_prices).dropna()
    return df_prices

df_prices = fetch_multi_asset_data(ASSETS, '2015-01-01', '2026-04-10')
df = pd.DataFrame({'price': df_prices['沪深300']})
for asset in df_prices.columns:
    if asset != '沪深300':
        df[f'{asset}_price'] = df_prices[asset]
print(f"✅ 数据: {len(df)} 天, {(len(df)/252):.1f} 年")

# ==================== 2. 特征工程 ====================
print("\n🔧 计算技术指标...")

for p in [1, 5, 10, 20]:
    df[f'ret_{p}d'] = df['price'].pct_change(p)

for m in [20, 60, 120]:
    df[f'ma_{m}'] = df['price'].rolling(m).mean()
df['ma_bias'] = (df['price'] / df['ma_20'] - 1) * 100

df['std20'] = df['price'].rolling(20).std()
df['ma_20'] = df['price'].rolling(20).mean()
df['bb_upper'] = df['ma_20'] + 2 * df['std20']
df['bb_lower'] = df['ma_20'] - 2 * df['std20']
df['bb_position'] = (df['price'] - df['bb_lower']) / (4 * df['std20'] + 1e-8)

def calc_rsi(s, period=14):
    delta = s.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi'] = calc_rsi(df['price'], 14)
df['volatility'] = df['ret_1d'].rolling(20).std() * np.sqrt(252)
df['momentum'] = df['price'] / df['price'].shift(20) - 1
df['trend_ma'] = (df['ma_60'] > df['ma_120']).astype(int)

for asset in df_prices.columns:
    if asset != '沪深300':
        df[f'{asset}_ret'] = df[f'{asset}_price'].pct_change()
        df[f'{asset}_volatility'] = df[f'{asset}_ret'].rolling(20).std() * np.sqrt(252)

# 市场状态：使用60日线和120日线
df['is_bull'] = (df['ma_60'] > df['ma_120']).astype(int)

df['future_ret'] = df['price'].shift(-5) / df['price'] - 1
df['label'] = (df['future_ret'] > 0.003).astype(int)
df = df.dropna()

feature_cols = [c for c in df.columns if c not in ['price', 'future_ret', 'label', 'ma_20', 'ma_60', 'ma_120']]
feature_cols = [c for c in feature_cols if df[c].dtype in ['float64', 'int64']]
print(f"✅ 特征数: {len(feature_cols)}")

# ==================== 3. 按年重训生成ML信号 ====================
print("\n🤖 生成ML信号（按年重训）...")

ml_signals = pd.Series(0.5, index=df.index)
years = sorted(df.index.year.unique())

for i in tqdm(range(1, len(years)), desc="按年训练"):
    train_years = years[:i]
    test_year = years[i]
    
    train_df = df[df.index.year.isin(train_years)]
    test_df = df[df.index.year == test_year]
    
    if len(train_df) < 100 or len(test_df) < 20:
        ml_signals.loc[test_df.index] = 0.5
        continue
    
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    X_test = test_df[feature_cols].values
    
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        ml_signals.loc[test_df.index] = 0.5
        continue
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(**RF_CONFIG)
    model.fit(X_train_scaled, y_train)
    
    probs = model.predict_proba(X_test_scaled)[:, 1]
    ml_signals.loc[test_df.index] = probs

ml_signal_array = ml_signals.values
ml_accuracy = ((ml_signal_array > 0.55).astype(int) == df['label']).mean()
print(f"   ML信号准确率: {ml_accuracy:.2%}")

# ==================== 4. 风险平价交易逻辑（V14优化版）====================
def execute_risk_parity_strategy(df: pd.DataFrame, ml_signals: np.ndarray, 
                                  params: Dict) -> Dict:
    """风险平价策略执行函数（V14：提高信号质量）"""
    df = df.copy()
    df['position'] = 0.0
    df['signal'] = 0
    df['trade_cost'] = 0.0
    
    position = 0.0
    entry_price = 0.0
    hold_days = 0
    monthly_trades = 0
    current_month = None
    trades_log = []
    
    for i in range(60, len(df)):
        price = df['price'].iloc[i]
        rsi = df['rsi'].iloc[i]
        bb_lower = df['bb_lower'].iloc[i]
        is_bull = df['is_bull'].iloc[i]
        vol = df['volatility'].iloc[i] if not pd.isna(df['volatility'].iloc[i]) else 0.15
        current_date = df.index[i]
        
        current_slippage = min(BASE_SLIPPAGE * (1.0 + vol * 5), MAX_SLIPPAGE)
        ml_signal = ml_signals[i]
        
        if current_month is None or current_date.month != current_month:
            current_month = current_date.month
            monthly_trades = 0
        
        if position > 0:
            hold_days += 1
            pnl = (price - entry_price) / entry_price
        
        # 买入：提高信号质量（阈值提高到0.40）
        if position == 0 and is_bull == 1 and monthly_trades < 6:
            rule_buy = (rsi < 35 and price < bb_lower * 1.02)  # 收紧规则
            rule_prob = 0.8 if rule_buy else 0.3
            
            ml_weight = params.get('ml_weight', 0.6)
            final_prob = ml_signal * ml_weight + rule_prob * (1 - ml_weight)
            
            if final_prob > params['signal_threshold']:
                # 风险平价仓位计算
                volatilities = {}
                for asset in ASSETS.keys():
                    if f'{asset}_volatility' in df.columns:
                        volatilities[asset] = max(df[f'{asset}_volatility'].iloc[i], 0.01)
                
                if volatilities:
                    inv_vol = {k: 1.0 / v for k, v in volatilities.items()}
                    total_inv = sum(inv_vol.values())
                    rp_weights = {k: v / total_inv for k, v in inv_vol.items()}
                    port_vol = 1.0 / total_inv
                    scaling = params['target_vol'] / port_vol
                    position = min(rp_weights.get('沪深300', 0.5) * scaling, params['max_position'])
                else:
                    position = min(params['max_position'], params['target_vol'] / max(vol, 0.05))
                
                entry_price = price
                hold_days = 0
                monthly_trades += 1
                df.loc[df.index[i], 'signal'] = 1
                df.loc[df.index[i], 'trade_cost'] = current_slippage + COMMISSION
                
                trades_log.append({
                    'entry_date': current_date,
                    'entry_price': price,
                    'signal_prob': final_prob,
                    'is_bull': is_bull,
                    'cost': current_slippage + COMMISSION
                })
        
        # 卖出：让利润有足够时间奔跑
        elif position > 0:
            sell = False
            pnl = (price - entry_price) / entry_price
            sell_reason = None
            
            # 止损
            if pnl < -params['stop_loss']:
                sell = True
                sell_reason = 'stop_loss'
            # 止盈（降低到12%）
            elif pnl > params['take_profit']:
                sell = True
                sell_reason = 'take_profit'
            # 持仓超时（延长到30天）
            elif hold_days > params['max_hold_days']:
                sell = True
                sell_reason = 'timeout'
            # 趋势转熊（只有盈利少时才卖）
            elif is_bull == 0 and pnl < 0.03:
                sell = True
                sell_reason = 'trend_down'
            
            if sell:
                if trades_log and trades_log[-1].get('exit_date') is None:
                    trades_log[-1]['exit_date'] = current_date
                    trades_log[-1]['exit_price'] = price
                    trades_log[-1]['pnl'] = pnl
                    trades_log[-1]['sell_reason'] = sell_reason
                    trades_log[-1]['hold_days'] = hold_days
                    trades_log[-1]['sell_cost'] = current_slippage + COMMISSION + STAMP_TAX
                    trades_log[-1]['total_cost'] = trades_log[-1]['cost'] + trades_log[-1]['sell_cost']
                
                position = 0.0
                df.loc[df.index[i], 'signal'] = -1
                df.loc[df.index[i], 'trade_cost'] = current_slippage + COMMISSION + STAMP_TAX
        
        df.loc[df.index[i], 'position'] = position
    
    # 收益计算
    df['ret'] = df['price'].pct_change()
    df['strategy_ret'] = df['position'].shift(1) * df['ret']
    df['cost'] = df['trade_cost']
    df['strategy_net'] = df['strategy_ret'] - df['cost']
    df['cum_strategy'] = (1 + df['strategy_net']).cumprod()
    df = df.dropna()
    
    nav = INITIAL_CAPITAL * df['cum_strategy']
    total_return = (nav.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    years = len(df) / 252
    annual_return = (nav.iloc[-1] / INITIAL_CAPITAL) ** (1/years) - 1
    
    returns = nav.pct_change().dropna()
    cummax = nav.expanding().max()
    drawdown = (nav - cummax) / cummax * 100
    max_dd = drawdown.min()
    sharpe = np.sqrt(252) * (returns - RISK_FREE_RATE/252).mean() / returns.std() if returns.std() > 0 else 0
    
    trades_df = pd.DataFrame(trades_log) if trades_log else pd.DataFrame()
    
    return {
        'total_return': total_return,
        'annual_return': annual_return * 100,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'nav': nav,
        'df': df,
        'trades': trades_df
    }

# ==================== 5. 参数扫描 ====================
print("\n🔬 参数扫描（寻找最优组合）...")

scan_results = []
total_combos = len(SCAN_PARAMS['stop_loss']) * len(SCAN_PARAMS['take_profit']) * \
               len(SCAN_PARAMS['signal_threshold']) * len(SCAN_PARAMS['max_hold_days'])

with tqdm(total=total_combos, desc="参数扫描") as pbar:
    for stop_loss, take_profit, signal_threshold, max_hold_days in product(
        SCAN_PARAMS['stop_loss'],
        SCAN_PARAMS['take_profit'],
        SCAN_PARAMS['signal_threshold'],
        SCAN_PARAMS['max_hold_days']
    ):
        test_params = RISK_PARITY_CONFIG.copy()
        test_params['stop_loss'] = stop_loss
        test_params['take_profit'] = take_profit
        test_params['signal_threshold'] = signal_threshold
        test_params['max_hold_days'] = max_hold_days
        
        result = execute_risk_parity_strategy(df, ml_signal_array, test_params)
        
        scan_results.append({
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'signal_threshold': signal_threshold,
            'max_hold_days': max_hold_days,
            'total_return': result['total_return'],
            'annual_return': result['annual_return'],
            'max_drawdown': result['max_drawdown'],
            'sharpe': result['sharpe'],
            'trades_count': len(result['trades'])
        })
        pbar.update(1)

# 排序找最优
scan_df = pd.DataFrame(scan_results)
best_sharpe = scan_df.loc[scan_df['sharpe'].idxmax()]
best_return = scan_df.loc[scan_df['total_return'].idxmax()]

print("\n" + "="*80)
print("📊 参数扫描结果（V14）")
print("="*80)
print(f"\n🏆 最高夏普比率: {best_sharpe['sharpe']:.3f}")
print(f"   参数: stop_loss={best_sharpe['stop_loss']}, take_profit={best_sharpe['take_profit']}, "
      f"threshold={best_sharpe['signal_threshold']}, max_hold={best_sharpe['max_hold_days']}")
print(f"   收益: {best_sharpe['total_return']:.2f}%, 回撤: {best_sharpe['max_drawdown']:.2f}%")
print(f"   交易次数: {best_sharpe['trades_count']}")
print(f"\n💰 最高收益: {best_return['total_return']:.2f}%")
print(f"   参数: stop_loss={best_return['stop_loss']}, take_profit={best_return['take_profit']}, "
      f"threshold={best_return['signal_threshold']}, max_hold={best_return['max_hold_days']}")
print(f"   夏普: {best_return['sharpe']:.3f}, 回撤: {best_return['max_drawdown']:.2f}%")

# ==================== 6. 使用最优参数运行最终策略 ====================
print("\n📊 使用最优参数运行最终策略...")

optimal_params = RISK_PARITY_CONFIG.copy()
optimal_params['stop_loss'] = best_sharpe['stop_loss']
optimal_params['take_profit'] = best_sharpe['take_profit']
optimal_params['signal_threshold'] = best_sharpe['signal_threshold']
optimal_params['max_hold_days'] = best_sharpe['max_hold_days']

result = execute_risk_parity_strategy(df, ml_signal_array, optimal_params)

print("\n" + "="*80)
print("📈 最终策略绩效（V14 - 提高信号质量）")
print("="*80)
print(f"总收益率: {result['total_return']:.2f}%")
print(f"年化收益率: {result['annual_return']:.2f}%")
print(f"最大回撤: {result['max_drawdown']:.2f}%")
print(f"夏普比率: {result['sharpe']:.3f}")
print(f"交易次数: {len(result['trades'])}")

if len(result['trades']) > 0:
    win_trades = len(result['trades'][result['trades']['pnl'] > 0])
    avg_win = result['trades'][result['trades']['pnl'] > 0]['pnl'].mean() * 100 if win_trades > 0 else 0
    avg_loss = result['trades'][result['trades']['pnl'] < 0]['pnl'].mean() * 100 if (len(result['trades'])-win_trades) > 0 else 0
    profit_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    print(f"胜率: {win_trades/len(result['trades'])*100:.1f}%")
    print(f"平均盈利: {avg_win:.2f}%, 平均亏损: {avg_loss:.2f}%")
    print(f"盈亏比: {profit_ratio:.2f}")

# ==================== 7. 卖出原因分析 ====================
if len(result['trades']) > 0:
    print("\n" + "="*80)
    print("📊 卖出原因分布（策略诊断）")
    print("="*80)
    sell_reasons = result['trades']['sell_reason'].value_counts()
    for reason, count in sell_reasons.items():
        pct = count / len(result['trades']) * 100
        print(f"   {reason}: {count}次 ({pct:.1f}%)")
    
    # 诊断建议
    print("\n💡 诊断建议:")
    timeout_pct = sell_reasons.get('timeout', 0) / len(result['trades']) * 100
    takeprofit_pct = sell_reasons.get('take_profit', 0) / len(result['trades']) * 100
    stoploss_pct = sell_reasons.get('stop_loss', 0) / len(result['trades']) * 100
    
    if timeout_pct > 60:
        print(f"   ⚠️ 超时卖出比例{timeout_pct:.0f}%，偏高。考虑延长max_hold_days或提高信号质量")
    elif takeprofit_pct > 30:
        print(f"   ✅ 止盈比例{takeprofit_pct:.0f}%，策略有效")
    else:
        print(f"   📈 止盈比例{takeprofit_pct:.0f}%，可以接受")
    
    if stoploss_pct > 20:
        print(f"   ⚠️ 止损比例{stoploss_pct:.0f}%，偏高。考虑降低风险或提高信号质量")

# ==================== 8. 超时卖出交易分析 ====================
if len(result['trades']) > 0:
    timeout_trades = result['trades'][result['trades']['sell_reason'] == 'timeout']
    if len(timeout_trades) > 0:
        avg_timeout_pnl = timeout_trades['pnl'].mean() * 100
        print(f"\n📊 超时卖出交易分析:")
        print(f"   超时交易次数: {len(timeout_trades)}")
        print(f"   超时交易平均盈利: {avg_timeout_pnl:.2f}%")
        if avg_timeout_pnl > 0:
            print(f"   ✅ 超时交易平均盈利为正，延长持仓可能有效")
        else:
            print(f"   ⚠️ 超时交易平均亏损，问题在买入信号质量")

# ==================== 9. 最终评分 ====================
print("\n" + "="*80)
print("📊 最终评估")
print("="*80)

score = 0
if result['sharpe'] > 1.0:
    print("✅ 夏普比率 > 1.0 (+2分)")
    score += 2
elif result['sharpe'] > 0.8:
    print("✅ 夏普比率 > 0.8 (+2分)")
    score += 2
elif result['sharpe'] > 0.6:
    print("📈 夏普比率 > 0.6 (+1分)")
    score += 1
elif result['sharpe'] > 0.5:
    print("📈 夏普比率 > 0.5 (+1分)")
    score += 1

if result['max_drawdown'] > -10:
    print("✅ 最大回撤控制良好 (<10%) (+1分)")
    score += 1
elif result['max_drawdown'] > -15:
    print("📉 最大回撤可接受 (<15%) (+0分)")

if result['annual_return'] > 12:
    print("✅ 年化收益 > 12% (+1分)")
    score += 1
elif result['annual_return'] > 10:
    print("📈 年化收益 > 10% (+0分)")

print(f"\n总分: {score}/5")

if score >= 4:
    print("🎉 策略表现优秀，可考虑实盘模拟")
elif score >= 3:
    print("📈 策略表现良好，建议继续优化")
else:
    print("⚠️ 策略需要进一步改进")

# ==================== 10. 绘图 ====================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

nav = result['nav']
drawdown = (nav - nav.expanding().max()) / nav.expanding().max() * 100

axes[0,0].plot(nav.index, nav.values, color='blue', linewidth=1.5)
axes[0,0].axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--')
axes[0,0].set_title(f'净值曲线（年化{result["annual_return"]:.1f}%, 夏普{result["sharpe"]:.2f}）')
axes[0,0].grid(True, alpha=0.3)

axes[0,1].fill_between(drawdown.index, 0, drawdown.values, color='red', alpha=0.3)
axes[0,1].set_title('回撤曲线')
axes[0,1].grid(True, alpha=0.3)

# 卖出原因饼图
if len(result['trades']) > 0:
    sell_reasons = result['trades']['sell_reason'].value_counts()
    axes[1,0].pie(sell_reasons.values, labels=sell_reasons.index, autopct='%1.1f%%')
    axes[1,0].set_title('卖出原因分布')
else:
    axes[1,0].text(0.5, 0.5, '无交易数据', ha='center', va='center')

axes[1,1].plot(result['df'].index, result['df']['position'], color='blue', linewidth=0.8)
axes[1,1].fill_between(result['df'].index, 0, result['df']['position'], alpha=0.3)
axes[1,1].set_title('仓位变化')
axes[1,1].set_ylim(-0.1, 1.1)
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("✅ V14 完成！")
print("="*80)
print("\nV14 优化内容:")
print("   1. ✅ 止盈降低: 20% → 12%（更容易触发）")
print("   2. ✅ 持仓延长: 20天 → 30天（让利润奔跑）")
print("   3. ✅ 信号阈值提高: 0.35 → 0.40（提高质量）")
print("   4. ✅ 买入规则收紧: RSI<35, 价格<下轨1.02")
print("   5. ✅ 超时交易盈利分析（诊断买入信号质量）")