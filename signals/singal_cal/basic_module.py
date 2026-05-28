import os
import sys
import logging

from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger('scan_signals')


def _ema(arr, span):
    """向量化EMA计算"""
    alpha = 2.0 / (span + 1)
    out = np.empty_like(arr, dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _sma(arr, n, m=1):
    """通达信SMA: SMA(X,N,M) = M/N * X + (N-M)/N * prev"""
    alpha = m / n
    out = np.empty_like(arr, dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _rolling_mean(arr, window):
    """快速滚动均值，返回与 arr 等长数组"""
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    cs = np.cumsum(arr)
    for i in range(n):
        start = max(0, i - window + 1)
        if start == 0:
            out[i] = cs[i] / (i + 1)
        else:
            out[i] = (cs[i] - cs[start - 1]) / window
    return out


def _rolling_max(arr, window):
    """滚动最大值"""
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - window + 1)
        out[i] = np.max(arr[start:i + 1])
    return out


def _rolling_min(arr, window):
    """滚动最小值"""
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - window + 1)
        out[i] = np.min(arr[start:i + 1])
    return out


def calculate_kdj(close_arr, high_arr, low_arr, n=9, m1=3, m2=3):
    """计算KDJ指标 - 向量化版本"""
    length = len(close_arr)
    high_n = _rolling_max(high_arr, n)
    low_n = _rolling_min(low_arr, n)

    denom = high_n - low_n
    rsv = np.where(denom != 0, (close_arr - low_n) / denom * 100, 50.0)

    k_arr = _sma(rsv, m1, 1)
    d_arr = _sma(k_arr, m2, 1)
    j_arr = 3 * k_arr - 2 * d_arr

    return float(k_arr[-1]), float(d_arr[-1]), float(j_arr[-1])


def calculate_知行多空线_arr(close_arr: np.ndarray, require_min_days: int = 114) -> np.ndarray:
    """计算知行多空线数组 - 向量化版本"""
    n = len(close_arr)
    if n < require_min_days:
        return np.array([])

    ma14 = _rolling_mean(close_arr, 14)
    ma28 = _rolling_mean(close_arr, 28)
    ma57 = _rolling_mean(close_arr, 57)
    ma114 = _rolling_mean(close_arr, 114)

    return (ma14 + ma28 + ma57 + ma114) / 4


def calculate_知行短期趋势线_arr(close_arr: np.ndarray) -> np.ndarray:
    """计算知行短期趋势线数组（EMA10的EMA10）- 向量化版本"""
    ema10_first = _ema(close_arr, 10)
    ema10_second = _ema(ema10_first, 10)
    return ema10_second


def calculate_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """计算技术指标 - 性能优化版本"""
    code = df['code'].values[0]
    close_arr = np.asarray(df['close'].values, dtype=np.float64)
    open_arr = np.asarray(df['open'].values, dtype=np.float64)
    high_arr = np.asarray(df['high'].values, dtype=np.float64)
    low_arr = np.asarray(df['low'].values, dtype=np.float64)
    volume_arr = np.asarray(df['volume'].values, dtype=np.float64)
    volume_arr = np.nan_to_num(volume_arr, nan=0)

    close = close_arr[-1]
    open_ = open_arr[-1]
    high = high_arr[-1]
    low = low_arr[-1]
    volume = volume_arr[-1]

    n = len(close_arr)

    prev_close = close_arr[-2] if n >= 2 else close_arr[-1]
    涨幅 = (close - prev_close) / prev_close * 100 if prev_close != 0 else 0
    振幅 = (high - low) / prev_close * 100 if prev_close != 0 else 0

    波幅 = float(np.mean(np.abs(high_arr[-30:] - low_arr[-30:])))
    波动率 = 波幅 / prev_close * 100 if prev_close != 0 else 0
    涨跌幅 = 涨幅
    大长阳 = close > open_ and 涨跌幅 > 波动率 * 1.5 and 涨跌幅 > 2
    大长阴 = close < open_ and abs(涨跌幅) > 波动率 * 1.1 and abs(涨跌幅) > 2
    参考成交量 = volume_arr[-2] if volume <= volume / 8 else volume_arr[-1] if n >= 2 else volume
    关键K = (close > close_arr[-2] and volume > 参考成交量 * 1.8 and 大长阳 and
             volume > np.mean(volume_arr[-40:])) if n >= 40 else False
    暴力K = (close > close_arr[-2] and volume > 参考成交量 * 1.8 and 涨跌幅 > 4 and
             (high - max(close, open_)) <= (high - open_) / 4 and
             volume > np.mean(volume_arr[-60:])) if n >= 60 else False

    # 均线 - 直接 numpy slice mean
    ma5 = float(np.mean(close_arr[-5:])) if n >= 5 else float(close)
    ma10 = float(np.mean(close_arr[-10:])) if n >= 10 else float(close)
    ma14 = float(np.mean(close_arr[-14:])) if n >= 14 else float(close)
    ma20 = float(np.mean(close_arr[-20:])) if n >= 20 else float(close)
    ma28 = float(np.mean(close_arr[-28:])) if n >= 28 else float(close)
    ma30 = float(np.mean(close_arr[-30:])) if n >= 30 else float(close)
    ma50 = float(np.mean(close_arr[-50:])) if n >= 50 else float(close)
    ma57 = float(np.mean(close_arr[-57:])) if n >= 57 else float(close)
    ma60 = float(np.mean(close_arr[-60:])) if n >= 60 else float(close)
    ma114 = float(np.mean(close_arr[-114:])) if n >= 114 else float(close)
    ma3 = float(np.mean(close_arr[-3:])) if n >= 3 else float(close)
    ma6 = float(np.mean(close_arr[-6:])) if n >= 6 else float(close)
    ma12 = float(np.mean(close_arr[-12:])) if n >= 12 else float(close)
    ma24 = float(np.mean(close_arr[-24:])) if n >= 24 else float(close)

    # 成交量均线
    vol_ma5 = float(np.mean(volume_arr[-5:])) if n >= 5 else float(volume)
    vol_ma10 = float(np.mean(volume_arr[-10:])) if n >= 10 else float(volume)
    vol_ma20 = float(np.mean(volume_arr[-20:])) if n >= 20 else float(volume)
    vol_ma60 = float(np.mean(volume_arr[-60:])) if n >= 60 else float(volume)

    # MACD DIF - 向量化EMA
    ema12_arr = _ema(close_arr, 12)
    ema26_arr = _ema(close_arr, 26)
    dif_arr = ema12_arr - ema26_arr
    dif = float(dif_arr[-1])

    # 知行短期趋势线 (EMA10 of EMA10)
    ema10_arr = _ema(close_arr, 10)
    知行短期趋势线 = float(_ema(ema10_arr, 10)[-1])

    # 知行多空线
    知行多空线 = (ma14 + ma28 + ma57 + ma114) / 4

    # DEA
    dea = 0

    # KDJ
    k, d, j = calculate_kdj(close_arr, high_arr, low_arr)

    # RSI - 向量化
    def _calc_rsi(period):
        if n < period + 1:
            return 50.0
        changes = np.diff(close_arr[-(period + 1):])
        gains = np.maximum(changes, 0)
        losses_v = np.maximum(-changes, 0)
        avg_g = np.mean(gains) if len(gains) > 0 else 0
        avg_l = np.mean(losses_v) if len(losses_v) > 0 else 0.001
        if avg_l == 0:
            avg_l = 0.001
        rs = avg_g / avg_l
        return float(100 - (100 / (1 + rs)))

    rsi1 = _calc_rsi(14)
    rsi2 = _calc_rsi(14)
    rsi3 = _calc_rsi(28)
    rsi4 = _calc_rsi(57)

    # BBI
    bbi = (ma3 + ma6 + ma12 + ma24) / 4

    if n >= 21:
        ma3_20 = float(np.mean(close_arr[-23:-20])) if n >= 23 else float(close)
        ma6_20 = float(np.mean(close_arr[-26:-20])) if n >= 26 else float(close)
        ma12_20 = float(np.mean(close_arr[-32:-20])) if n >= 32 else float(close)
        ma24_20 = float(np.mean(close_arr[-44:-20])) if n >= 44 else float(close)
        前20日BBI = (ma3_20 + ma6_20 + ma12_20 + ma24_20) / 4
    else:
        前20日BBI = bbi

    return {
        'code': code,
        'open_arr': open_arr,
        'high_arr': high_arr,
        'low_arr': low_arr,
        'close_arr': close_arr,
        'volume_arr': volume_arr,
        'open': float(open_),
        'high': float(high),
        'low': float(low),
        'close': float(close),
        'volume': float(volume),
        'prev_close': float(prev_close),
        '涨幅': float(涨幅),
        '振幅': float(振幅),
        '波幅': float(波幅),
        '波动率': float(波动率),
        '大长阳': float(大长阳),
        '大长阴': float(大长阴),
        '参考成交量': float(参考成交量),
        '关键K': float(关键K),
        '暴力K': float(暴力K),
        'ma5': ma5, 'ma10': ma10, 'ma14': ma14, 'ma20': ma20,
        'ma28': ma28, 'ma30': ma30, 'ma50': ma50, 'ma57': ma57,
        'ma60': ma60, 'ma114': ma114, 'ma3': ma3, 'ma6': ma6,
        'ma12': ma12, 'ma24': ma24,
        'dif': dif,
        'dif_arr': dif_arr,
        'dea': float(dea),
        'k': float(k), 'd': float(d), 'j': float(j),
        'rsi1': rsi1, 'rsi2': rsi2, 'rsi3': rsi3, 'rsi4': rsi4,
        'bbi': bbi,
        '前20日BBI': 前20日BBI,
        '知行短期趋势线': 知行短期趋势线,
        '知行多空线': 知行多空线,
        'vol_ma5': vol_ma5, 'vol_ma10': vol_ma10,
        'vol_ma20': vol_ma20, 'vol_ma60': vol_ma60,
    }
