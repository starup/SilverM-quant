#!/usr/bin/env python3
"""
轻量级回测引擎 - 即时计算信号，逐日模拟交易

不依赖 daily_signals 表，直接调用 singal_cal 模块实时计算。
对日线策略精度误差约 0.1-0.2%，方向性结论一致。

使用:
    python backtest/strategy_backtest/lite_backtest.py --mode resonance --limit 50
    python backtest/strategy_backtest/lite_backtest.py --mode single --strategy b1 --limit 100
"""
import os
import sys
import json
import argparse
import time
import duckdb
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'signals', 'singal_cal'))

from basic_module import calculate_indicators
from B1_strategy_module import calculate_b1_score
from B2_strategy_module import calculate_b2_score
from BLKB2_strategy_module import check_暴力K, check_倍量柱, check_J拐头向上
from SCB_strategy_module import calculate_scb_signal

DB_PATH = os.path.join(PROJECT_ROOT, 'data', 'Astock3.duckdb')

COMMISSION = 0.0003
STAMP_DUTY = 0.001
SLIPPAGE = 0.001


def get_buy_signals(indicators: dict, b1_threshold=8) -> List[str]:
    """根据指标判断哪些策略触发买入"""
    signals = []

    try:
        b1_score = calculate_b1_score(indicators)
        j_low = indicators.get('j', 100) < 13
        dif_pos = indicators.get('dif', -1) >= 0
        trend_ok = indicators.get('知行短期趋势线', 0) > indicators.get('知行多空线', 999)
        if j_low and dif_pos and trend_ok and b1_score >= b1_threshold:
            signals.append('b1')
    except Exception:
        pass

    try:
        b2_score = calculate_b2_score(indicators)
        dif_pos = indicators.get('dif', -1) >= 0
        trend_ok = indicators.get('知行短期趋势线', 0) > indicators.get('知行多空线', 999)
        if dif_pos and trend_ok and b2_score >= 8:
            signals.append('b2')
    except Exception:
        pass

    try:
        blk = check_暴力K(indicators)
        trend_ok = indicators.get('知行短期趋势线', 0) > indicators.get('知行多空线', 999)
        if blk and trend_ok:
            signals.append('blk')
    except Exception:
        pass

    try:
        scb_score = calculate_scb_signal(indicators)
        if scb_score and scb_score > 0:
            signals.append('scb')
    except Exception:
        pass

    return signals


def should_sell(indicators: dict, entry_price: float, current_price: float, stop_loss: float = 0.03) -> Optional[str]:
    """判断是否卖出"""
    pnl = (current_price - entry_price) / entry_price
    if pnl < -stop_loss:
        return f"止损({stop_loss*100:.0f}%)"

    trend_line = indicators.get('知行多空线', 0)
    if current_price < trend_line * 0.98:
        return "跌破多空线"

    return None


class LiteBacktest:
    def __init__(
        self,
        initial_cash: float = 200000.0,
        max_positions: int = 5,
        max_single_pct: float = 0.20,
        stop_loss: float = 0.03,
        signal_mode: str = 'resonance',
        strategy_name: str = None,
        min_signals: int = 2,
        exclude_688: bool = True,
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.max_positions = max_positions
        self.max_single_pct = max_single_pct
        self.stop_loss = stop_loss
        self.signal_mode = signal_mode
        self.strategy_name = strategy_name
        self.min_signals = min_signals
        self.exclude_688 = exclude_688

        self.positions = {}
        self.trades = []
        self.daily_values = []
        self.daily_dates = []

    def _portfolio_value(self, prices: dict) -> float:
        value = self.cash
        for code, pos in self.positions.items():
            value += pos['shares'] * prices.get(code, pos['entry_price'])
        return value

    def run(self, stocks: List[str], start_date: str, end_date: str) -> dict:
        """运行回测"""
        from database.connection import get_connection
        conn = get_connection(DB_PATH)
        from_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
        to_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

        trade_dates = conn.execute(f"""
            SELECT DISTINCT trade_date FROM dwd_daily_price_qfq
            WHERE trade_date BETWEEN '{from_date}' AND '{to_date}'
            ORDER BY trade_date
        """).fetchall()
        trade_dates = [str(row[0]) for row in trade_dates]

        stock_data = {}
        for code in stocks:
            if self.exclude_688 and code.startswith('688'):
                continue
            ts_code = f"{code}.SH" if code.startswith('6') else f"{code}.SZ"
            df = conn.execute(f"""
                SELECT trade_date AS date, open, high, low, close, vol AS volume
                FROM dwd_daily_price_qfq
                WHERE ts_code = '{ts_code}'
                AND trade_date BETWEEN '{from_date}' AND '{to_date}'
                ORDER BY trade_date
            """).fetchdf()
            if df is not None and len(df) >= 60:
                df['code'] = code
                df['date'] = df['date'].astype(str)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(subset=['close'])
                stock_data[code] = df

        conn.close()

        if not stock_data:
            return {'status': 'error', 'error': '无有效数据'}

        print(f"回测: {len(stock_data)} 只股票, {from_date} ~ {to_date}, 模式={self.signal_mode}")

        for date_str in trade_dates:
            current_prices = {}
            buy_candidates = []

            for code, df in stock_data.items():
                idx = df[df['date'] == date_str].index
                if len(idx) == 0:
                    continue
                pos_in_df = df.index.get_loc(idx[0])
                if pos_in_df < 59:
                    continue

                window = df.iloc[max(0, pos_in_df - 149):pos_in_df + 1].copy()
                window = window.rename(columns={'date': 'date'})
                current_prices[code] = float(window['close'].iloc[-1])

                try:
                    indicators = calculate_indicators(window)
                except Exception:
                    continue

                if code in self.positions:
                    sell_reason = should_sell(indicators, self.positions[code]['entry_price'], current_prices[code], self.stop_loss)
                    if sell_reason:
                        self._sell(code, current_prices[code], date_str, sell_reason)
                else:
                    signals = get_buy_signals(indicators)
                    if self.signal_mode == 'resonance' and len(signals) >= self.min_signals:
                        buy_candidates.append((code, signals, current_prices[code]))
                    elif self.signal_mode == 'single' and self.strategy_name in signals:
                        buy_candidates.append((code, signals, current_prices[code]))

            buy_candidates.sort(key=lambda x: len(x[1]), reverse=True)
            for code, signals, price in buy_candidates:
                if len(self.positions) >= self.max_positions:
                    break
                self._buy(code, price, date_str, signals)

            self.daily_values.append(self._portfolio_value(current_prices))
            self.daily_dates.append(date_str)

        final_prices = {}
        for code, df in stock_data.items():
            if len(df) > 0:
                final_prices[code] = float(df['close'].iloc[-1])
        final_value = self._portfolio_value(final_prices)

        return self._build_result(final_value, trade_dates)

    def _buy(self, code: str, price: float, date: str, signals: list):
        max_amount = self._portfolio_value({}) * self.max_single_pct
        available = min(self.cash * 0.95, max_amount)
        shares = int(available / price / 100) * 100
        if shares < 100:
            return

        cost = shares * price * (1 + COMMISSION + SLIPPAGE)
        if cost > self.cash:
            return

        self.cash -= cost
        self.positions[code] = {
            'shares': shares,
            'entry_price': price,
            'entry_date': date,
            'signals': signals,
            'cost': cost,
        }
        self.trades.append({
            'action': 'BUY', 'code': code, 'date': date,
            'price': price, 'shares': shares, 'signals': signals,
        })

    def _sell(self, code: str, price: float, date: str, reason: str):
        pos = self.positions.pop(code)
        proceeds = pos['shares'] * price * (1 - COMMISSION - STAMP_DUTY - SLIPPAGE)
        self.cash += proceeds
        pnl = proceeds - pos['cost']
        pnl_pct = pnl / pos['cost'] * 100

        self.trades.append({
            'action': 'SELL', 'code': code, 'date': date,
            'price': price, 'shares': pos['shares'], 'reason': reason,
            'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2),
            'entry_date': pos['entry_date'], 'entry_price': pos['entry_price'],
            'signals': pos['signals'],
        })

    def _build_result(self, final_value: float, trade_dates: list) -> dict:
        total_return = (final_value - self.initial_cash) / self.initial_cash
        days = len(trade_dates)
        annual_return = (1 + total_return) ** (252 / max(days, 1)) - 1 if total_return > -1 else -1

        sell_trades = [t for t in self.trades if t['action'] == 'SELL']
        wins = [t for t in sell_trades if t['pnl'] >= 0]
        losses = [t for t in sell_trades if t['pnl'] < 0]
        win_rate = len(wins) / len(sell_trades) * 100 if sell_trades else 0
        avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0

        max_drawdown = 0
        peak = self.initial_cash
        for v in self.daily_values:
            peak = max(peak, v)
            dd = (peak - v) / peak
            max_drawdown = max(max_drawdown, dd)

        strategy_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0})
        for t in sell_trades:
            for sig in t.get('signals', ['unknown']):
                if t['pnl'] >= 0:
                    strategy_stats[sig]['wins'] += 1
                else:
                    strategy_stats[sig]['losses'] += 1
                strategy_stats[sig]['total_pnl'] += t['pnl']

        strat_summary = {}
        for s, stats in strategy_stats.items():
            total = stats['wins'] + stats['losses']
            strat_summary[s] = {
                'total': total,
                'win_rate': round(stats['wins'] / total * 100, 1) if total > 0 else 0,
                'total_pnl': round(stats['total_pnl'], 2),
            }

        return {
            'status': 'success',
            'initial_cash': self.initial_cash,
            'final_value': round(final_value, 2),
            'total_return_pct': round(total_return * 100, 2),
            'annual_return_pct': round(annual_return * 100, 2),
            'max_drawdown_pct': round(max_drawdown * 100, 2),
            'total_buy_trades': len([t for t in self.trades if t['action'] == 'BUY']),
            'total_sell_trades': len(sell_trades),
            'win_rate': round(win_rate, 1),
            'avg_win_pct': round(avg_win, 2),
            'avg_loss_pct': round(avg_loss, 2),
            'strategy_stats': strat_summary,
            'open_positions': len(self.positions),
            'days': days,
            'trades': self.trades,
            'daily_values': self.daily_values,
            'daily_dates': self.daily_dates,
        }


def main():
    parser = argparse.ArgumentParser(description='轻量级信号回测')
    parser.add_argument('--start', default='20250101')
    parser.add_argument('--end', default='20260527')
    parser.add_argument('--cash', type=float, default=200000)
    parser.add_argument('--mode', choices=['resonance', 'single'], default='resonance')
    parser.add_argument('--strategy', default=None, help='单策略: b1/b2/blk/scb')
    parser.add_argument('--min-signals', type=int, default=2)
    parser.add_argument('--max-positions', type=int, default=5)
    parser.add_argument('--stop-loss', type=float, default=0.03)
    parser.add_argument('--limit', type=int, default=100, help='回测股票数量')
    parser.add_argument('--output', default=None, help='输出 JSON 文件路径')
    args = parser.parse_args()

    from database.connection import get_connection
    conn = get_connection(DB_PATH)
    stocks = conn.execute(f"""
        SELECT symbol FROM dwd_stock_info
        WHERE list_status = 'L'
        ORDER BY RANDOM()
        LIMIT {args.limit}
    """).fetchall()
    stock_codes = [row[0] for row in stocks]
    conn.close()

    bt = LiteBacktest(
        initial_cash=args.cash,
        max_positions=args.max_positions,
        stop_loss=args.stop_loss,
        signal_mode=args.mode,
        strategy_name=args.strategy,
        min_signals=args.min_signals,
    )

    start_time = time.time()
    result = bt.run(stock_codes, args.start, args.end)
    elapsed = time.time() - start_time

    if result['status'] == 'success':
        print(f"\n{'='*50}")
        print(f"轻量级回测结果 (耗时 {elapsed:.1f}秒)")
        print(f"{'='*50}")
        print(f"初始资金:     {result['initial_cash']:,.0f}")
        print(f"最终价值:     {result['final_value']:,.0f}")
        print(f"总收益率:     {result['total_return_pct']:.2f}%")
        print(f"年化收益率:   {result['annual_return_pct']:.2f}%")
        print(f"最大回撤:     {result['max_drawdown_pct']:.2f}%")
        print(f"{'─'*50}")
        print(f"买入次数:     {result['total_buy_trades']}")
        print(f"卖出次数:     {result['total_sell_trades']}")
        print(f"胜率:         {result['win_rate']:.1f}%")
        print(f"平均盈利:     {result['avg_win_pct']:.2f}%")
        print(f"平均亏损:     {result['avg_loss_pct']:.2f}%")
        print(f"未平仓:       {result['open_positions']}")
        print(f"{'─'*50}")
        if result['strategy_stats']:
            print("策略表现:")
            for s, stats in sorted(result['strategy_stats'].items(), key=lambda x: x[1]['win_rate'], reverse=True):
                print(f"  {s:6s}: 胜率 {stats['win_rate']:.0f}%, 共{stats['total']}笔, 盈亏 {stats['total_pnl']:+.0f}")
        print(f"{'='*50}")

        if args.output:
            output = {k: v for k, v in result.items() if k not in ('daily_values', 'daily_dates', 'trades')}
            output['trades_summary'] = result['trades'][:50]
            with open(args.output, 'w') as f:
                json.dump(output, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n结果已保存到: {args.output}")
    else:
        print(f"回测失败: {result['error']}")


if __name__ == '__main__':
    main()
