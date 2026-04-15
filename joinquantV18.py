# ============================================
# 聚宽：随机森林策略（兼容版）
# ============================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ==================== 参数 ====================
STOCK = '510300.XSHG'

MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'random_state': 42,
    'n_jobs': 1
}

TRADE_PARAMS = {
    'signal_threshold': 0.55,
    'max_position': 0.95,
    'stop_loss': 0.05,
    'trailing_stop': 0.03,
    'max_hold_days': 20,
    'train_window_days': 500
}

SLIPPAGE = 0.0006
COMMISSION = 0.00015

# 全局变量
g = {
    'model': None,
    'scaler': None,
    'features': ['ret_1d', 'ret_5d', 'ret_20d', 'bias', 'vol_ratio', 'volatility', 'rsi', 'trend'],
    'entry_price': 0,
    'highest_price': 0,
    'hold_days': 0,
    'last_train_date': None,
    'daily_returns': []
}


# ==================== 特征工程 ====================

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def get_features(df):
    """计算特征 - 无未来数据泄露"""
    df = df.copy()
    
    # 收益率
    df['ret_1d'] = df['price'].pct_change(1)
    df['ret_5d'] = df['price'].pct_change(5)
    df['ret_20d'] = df['price'].pct_change(20)
    
    # 均线
    df['ma20'] = df['price'].rolling(20).mean()
    df['ma60'] = df['price'].rolling(60).mean()
    df['bias'] = (df['price'] / df['ma20'] - 1) * 100
    
    # 成交量
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # 波动率
    df['volatility'] = df['ret_1d'].rolling(20).std() * np.sqrt(252)
    
    # RSI
    df['rsi'] = calculate_rsi(df['price'], 14)
    
    # 趋势
    df['trend'] = (df['ma20'] > df['ma60']).astype(int)
    
    return df


def add_labels(df, forward_days=5, threshold=0.005):
    """添加标签"""
    df = df.copy()
    df['future_ret'] = df['price'].shift(-forward_days) / df['price'] - 1
    df['label'] = (df['future_ret'] > threshold).astype(int)
    return df.dropna()


def clean_array(X):
    """清理数组中的nan和inf - 兼容旧版numpy"""
    # 方法1：逐元素处理（兼容性最好）
    X = np.where(np.isnan(X), 0, X)
    X = np.where(np.isinf(X), 0, X)
    return X


# ==================== 训练模型 ====================

def train_model(df, current_date):
    """训练随机森林"""
    # 只用最近N天数据
    if len(df) > TRADE_PARAMS['train_window_days']:
        df = df.iloc[-TRADE_PARAMS['train_window_days']:]
    
    X = df[g['features']].values
    y = df['label'].values
    
    # 清理数据（兼容旧版numpy）
    X = clean_array(X)
    
    if len(X) < 100:
        return False
    
    # 按时间顺序划分
    split_idx = int(len(X) * 0.7)
    
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 训练
    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X_train_scaled, y_train)
    
    # 验证
    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    val_acc = accuracy_score(y_val, model.predict(X_val_scaled))
    
    # 保存
    g['model'] = model
    g['scaler'] = scaler
    g['last_train_date'] = current_date
    
    log.info(f"训练完成 | 窗口:{len(X)}天 | 训练集:{train_acc:.2%} | 验证集:{val_acc:.2%}")
    
    return True


# ==================== 交易信号 ====================

def trade_signal(context):
    """主交易逻辑"""
    current_date = context.current_dt.strftime('%Y-%m-%d')
    
    # 1. 获取历史数据
    df_price = get_price(STOCK, end_date=current_date, count=600,
                         frequency='daily', fields=['close', 'volume'], fq='pre')
    
    if df_price is None or len(df_price) < 100:
        return
    
    df = pd.DataFrame({
        'price': df_price['close'],
        'volume': df_price['volume']
    })
    
    # 2. 计算特征
    df = get_features(df)
    df = df.dropna()
    
    if len(df) < 50:
        return
    
    # 3. 判断是否需要训练
    need_train = False
    
    if g['last_train_date'] is None:
        need_train = True
    elif current_date[:7] != g['last_train_date'][:7]:
        need_train = True
    elif (pd.to_datetime(current_date) - pd.to_datetime(g['last_train_date'])).days > 60:
        need_train = True
    
    if need_train:
        # 训练时添加标签
        df_with_labels = add_labels(df.copy())
        if len(df_with_labels) >= 100:
            train_model(df_with_labels, current_date)
    
    if g['model'] is None:
        return
    
    # 4. 预测
    latest = df[g['features']].iloc[-1:].values
    latest = clean_array(latest)
    
    try:
        scaled = g['scaler'].transform(latest)
        prob = g['model'].predict_proba(scaled)[0, 1]
    except Exception as e:
        log.error(f"预测失败: {e}")
        return
    
    # 5. 获取当前价格
    current_data = get_current_data()
    if STOCK in current_data and current_data[STOCK].last_price:
        current_price = current_data[STOCK].last_price
    else:
        current_price = df['price'].iloc[-1]
    
    # 6. 持仓判断
    position = context.portfolio.positions.get(STOCK, None)
    has_position = position is not None and position.total_amount > 0
    
    trend = df['trend'].iloc[-1]
    
    # 7. 开仓
    if not has_position:
        if trend == 1 and prob > TRADE_PARAMS['signal_threshold']:
            target_value = context.portfolio.total_value * TRADE_PARAMS['max_position']
            order_target_value(STOCK, target_value)
            
            g['entry_price'] = current_price
            g['highest_price'] = current_price
            g['hold_days'] = 0
            
            log.info(f"【买入】价格:{current_price:.3f} | 信号:{prob:.3f} | 趋势:{trend}")
    
    # 8. 平仓
    else:
        g['hold_days'] += 1
        
        if current_price > g['highest_price']:
            g['highest_price'] = current_price
        
        pnl = (current_price - g['entry_price']) / g['entry_price']
        drawdown = (current_price - g['highest_price']) / g['highest_price']
        
        sell_reason = None
        
        if pnl < -TRADE_PARAMS['stop_loss']:
            sell_reason = f"止损(pnl:{pnl:.2%})"
        elif drawdown < -TRADE_PARAMS['trailing_stop']:
            sell_reason = f"回撤(dd:{drawdown:.2%})"
        elif g['hold_days'] >= TRADE_PARAMS['max_hold_days']:
            sell_reason = f"超期({g['hold_days']}天)"
        elif trend == 0:
            sell_reason = "趋势转弱"
        
        if sell_reason:
            order_target_value(STOCK, 0)
            log.info(f"【卖出】价格:{current_price:.3f} | 盈亏:{pnl:.2%} | {sell_reason}")
            
            g['entry_price'] = 0
            g['highest_price'] = 0
            g['hold_days'] = 0


# ==================== 收盘统计 ====================

def after_close(context):
    """收盘后记录"""
    current_value = context.portfolio.total_value
    
    if not hasattr(context, 'prev_value'):
        context.prev_value = current_value
        return
    
    daily_return = (current_value - context.prev_value) / context.prev_value
    g['daily_returns'].append(daily_return)
    context.prev_value = current_value
    
    # 每周输出
    if context.current_dt.weekday() == 4 and len(g['daily_returns']) > 0:
        total_return = (1 + np.array(g['daily_returns'])).prod() - 1
        
        # 计算夏普
        returns = np.array(g['daily_returns'][-20:])
        if len(returns) > 1 and returns.std() > 0:
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe = 0
        
        log.info("=" * 50)
        log.info(f"周报 | 累计收益:{total_return:+.2%} | 近20天夏普:{sharpe:.2f}")
        log.info("=" * 50)


# ==================== 初始化 ====================

def initialize(context):
    set_benchmark('000300.XSHG')
    set_slippage(FixedSlippage(SLIPPAGE))
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0.001,
        open_commission=COMMISSION, close_commission=COMMISSION,
        min_commission=5
    ), type='stock')
    set_option('use_real_price', True)
    
    run_daily(trade_signal, time='10:00')
    run_daily(after_close, time='15:10')
    
    log.info("=" * 50)
    log.info("随机森林策略启动")
    log.info(f"标的: {STOCK} | 阈值:{TRADE_PARAMS['signal_threshold']} | 止损:{TRADE_PARAMS['stop_loss']:.0%}")
    log.info("=" * 50)


def init(context):
    initialize(context)