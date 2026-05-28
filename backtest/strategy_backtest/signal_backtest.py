"""
基于 daily_signals 表的信号回测策略

不重新计算技术指标，直接读取 daily_signals 表中已有的信号结果。
让 backtrader 处理资金管理、手续费、滑点等仿真，信号判断由信号系统提供。
"""
import os
import sys
import backtrader as bt
import duckdb
import pandas as pd
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'Astock3.duckdb')


class SignalBasedStrategy(bt.Strategy):
    """
    基于 daily_signals 表的回测策略

    买入条件: daily_signals 中对应日期+代码有买入信号
    卖出条件: daily_signals 中对应日期+代码有卖出信号，或止损触发
    """

    params = (
        ('signal_mode', 'resonance'),   # resonance=多信号共振, single=单策略
        ('strategy_name', None),         # single 模式下指定策略名: b1, b2, blk, dl, dz30, scb, blkB2
        ('min_signals', 2),              # resonance 模式下最少信号数
        ('max_positions', 5),
        ('stop_loss_pct', 0.03),
        ('max_single_pct', 0.20),        # 单只最大仓位比例
        ('exclude_688', True),           # 排除科创板
        ('max_change_pct', 7.0),         # 最大涨幅过滤
        ('debug_mode', False),
    )

    def __init__(self):
        self.daily_values = []
        self.daily_dates = []
        self.trade_records = []
        self.pending_orders = {}
        self.entry_prices = {}
        self._signals_cache = {}
        self._load_signals()

    def _load_signals(self):
        """预加载所有信号数据到内存"""
        conn = duckdb.connect(DB_PATH, read_only=True)
        df = conn.execute("""
            SELECT date, code,
                   signal_buy_b1, signal_buy_b2, signal_buy_blk,
                   signal_buy_dl, signal_buy_dz30, signal_buy_scb, signal_buy_blkB2,
                   signal_s1_full, signal_跌破多空线, signal_止损,
                   change_pct, close
            FROM daily_signals
        """).fetchdf()
        conn.close()

        for _, row in df.iterrows():
            date_str = str(row['date'])
            code = row['code']
            key = (date_str, code)

            buy_signals = []
            signal_names = ['b1', 'b2', 'blk', 'dl', 'dz30', 'scb', 'blkB2']
            for i, name in enumerate(signal_names):
                if row.iloc[2 + i]:
                    buy_signals.append(name)

            sell_signals = []
            if row.get('signal_s1_full'):
                sell_signals.append('s1_full')
            if row.get('signal_跌破多空线'):
                sell_signals.append('跌破多空线')

            self._signals_cache[key] = {
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'change_pct': float(row['change_pct']) if row['change_pct'] else 0,
            }

    def _get_signal(self, date_str: str, code: str) -> dict:
        return self._signals_cache.get((date_str, code), {'buy_signals': [], 'sell_signals': [], 'change_pct': 0})

    def next(self):
        current_date = self.datas[0].datetime.datetime(0)
        self.daily_values.append(self.broker.getvalue())
        self.daily_dates.append(current_date)

        date_str = current_date.strftime('%Y-%m-%d')

        for data in self.datas:
            code = data._name
            position = self.getposition(data)
            signal = self._get_signal(date_str, code)

            if code in self.pending_orders:
                continue

            if not position.size:
                if self._should_buy(code, signal):
                    self._execute_buy(data)
            else:
                if self._should_sell(data, code, signal, position):
                    self._execute_sell(data)

    def _should_buy(self, code: str, signal: dict) -> bool:
        if self.params.exclude_688 and code.startswith('688'):
            return False

        if abs(signal['change_pct']) > self.params.max_change_pct:
            return False

        buy_signals = signal['buy_signals']

        if self.params.signal_mode == 'resonance':
            return len(buy_signals) >= self.params.min_signals
        else:
            return self.params.strategy_name in buy_signals

    def _should_sell(self, data, code: str, signal: dict, position) -> bool:
        if signal['sell_signals']:
            return True

        entry_price = self.entry_prices.get(code, data.close[0])
        current_price = data.close[0]
        pnl_pct = (current_price - entry_price) / entry_price
        if pnl_pct < -self.params.stop_loss_pct:
            return True

        return False

    def _execute_buy(self, data):
        current_positions = len([d for d in self.datas if self.getposition(d).size > 0])
        if current_positions >= self.params.max_positions:
            return

        price = data.close[0]
        total_value = self.broker.getvalue()
        max_amount = total_value * self.params.max_single_pct
        cash = self.broker.getcash()
        available = min(cash * 0.95, max_amount)

        if available < price * 100:
            return

        size = int(available / price / 100) * 100
        if size < 100:
            return

        order = self.buy(data=data, size=size)
        self.pending_orders[data._name] = order
        self.entry_prices[data._name] = price

    def _execute_sell(self, data):
        order = self.close(data=data)
        self.pending_orders[data._name] = order
        if data._name in self.entry_prices:
            del self.entry_prices[data._name]

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            data_name = order.data._name
            if data_name in self.pending_orders:
                del self.pending_orders[data_name]

            if order.status == order.Completed:
                current_date = self.datas[0].datetime.datetime(0)
                self.trade_records.append({
                    'date': current_date,
                    'action': 'BUY' if order.isbuy() else 'SELL',
                    'code': data_name,
                    'price': order.executed.price,
                    'size': order.executed.size,
                })

    def get_daily_values(self) -> List[float]:
        return self.daily_values

    def get_daily_dates(self) -> List[datetime]:
        return self.daily_dates


def run_signal_backtest(
    start_date: str = '20250101',
    end_date: str = '20260527',
    initial_cash: float = 200000.0,
    signal_mode: str = 'resonance',
    strategy_name: str = None,
    min_signals: int = 2,
    max_positions: int = 5,
    stock_limit: int = 200,
) -> dict:
    """
    运行基于信号表的回测

    Args:
        signal_mode: 'resonance' (多信号共振) 或 'single' (单策略)
        strategy_name: single模式下的策略名 (b1/b2/blk/dl/dz30/scb/blkB2)
        min_signals: resonance模式下最少信号数
        stock_limit: 最多加载多少只股票
    """
    conn = duckdb.connect(DB_PATH, read_only=True)

    from_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    to_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

    if signal_mode == 'resonance':
        stocks = conn.execute(f"""
            SELECT DISTINCT code FROM daily_signals
            WHERE date BETWEEN '{from_date}' AND '{to_date}'
            AND code NOT LIKE '688%'
            AND (CAST(signal_buy_b1 AS INT) + CAST(signal_buy_b2 AS INT) +
                 CAST(signal_buy_blk AS INT) + CAST(signal_buy_dl AS INT) +
                 CAST(signal_buy_dz30 AS INT) + CAST(signal_buy_scb AS INT) +
                 CAST(signal_buy_blkB2 AS INT)) >= {min_signals}
            LIMIT {stock_limit}
        """).fetchall()
    else:
        field = f"signal_buy_{strategy_name}"
        stocks = conn.execute(f"""
            SELECT DISTINCT code FROM daily_signals
            WHERE date BETWEEN '{from_date}' AND '{to_date}'
            AND code NOT LIKE '688%'
            AND {field} = true
            LIMIT {stock_limit}
        """).fetchall()

    stock_codes = [row[0] for row in stocks]

    if not stock_codes:
        conn.close()
        return {'status': 'error', 'error': '没有找到符合条件的股票'}

    cereb = bt.Cerebro()
    cereb.broker.setcash(initial_cash)
    cereb.broker.setcommission(commission=0.0003)

    cereb.addstrategy(
        SignalBasedStrategy,
        signal_mode=signal_mode,
        strategy_name=strategy_name,
        min_signals=min_signals,
        max_positions=max_positions,
    )

    valid_count = 0
    for code in stock_codes:
        ts_code = f"{code}.SH" if code.startswith('6') else f"{code}.SZ"
        df = conn.execute(f"""
            SELECT trade_date AS date, open, high, low, close, vol AS volume
            FROM dwd_daily_price_qfq
            WHERE ts_code = '{ts_code}'
            AND trade_date BETWEEN '{from_date}' AND '{to_date}'
            ORDER BY trade_date
        """).fetchdf()

        if df is None or len(df) < 30:
            continue

        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()

        if len(df) < 30:
            continue

        data = bt.feeds.PandasData(dataname=df, name=code)
        cereb.adddata(data)
        valid_count += 1

    conn.close()

    if valid_count == 0:
        return {'status': 'error', 'error': '没有有效数据'}

    print(f"回测: {valid_count} 只股票, {from_date} ~ {to_date}, 模式={signal_mode}")
    results = cereb.run()
    strat = results[0]

    final_value = cereb.broker.getvalue()
    total_return = (final_value - initial_cash) / initial_cash
    days = len(strat.daily_values)
    annual_return = (1 + total_return) ** (252 / max(days, 1)) - 1 if total_return > -1 else -1

    trades = strat.trade_records
    buy_trades = [t for t in trades if t['action'] == 'BUY']
    sell_trades = [t for t in trades if t['action'] == 'SELL']

    return {
        'status': 'success',
        'initial_cash': initial_cash,
        'final_value': round(final_value, 2),
        'total_return_pct': round(total_return * 100, 2),
        'annual_return_pct': round(annual_return * 100, 2),
        'total_trades': len(buy_trades),
        'stock_count': valid_count,
        'days': days,
        'trades': trades,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='基于信号表的回测')
    parser.add_argument('--start', default='20250101')
    parser.add_argument('--end', default='20260527')
    parser.add_argument('--cash', type=float, default=200000)
    parser.add_argument('--mode', choices=['resonance', 'single'], default='resonance')
    parser.add_argument('--strategy', default=None, help='单策略名: b1/b2/blk/dl/dz30/scb/blkB2')
    parser.add_argument('--min-signals', type=int, default=2)
    parser.add_argument('--max-positions', type=int, default=5)
    parser.add_argument('--limit', type=int, default=200)
    args = parser.parse_args()

    result = run_signal_backtest(
        start_date=args.start,
        end_date=args.end,
        initial_cash=args.cash,
        signal_mode=args.mode,
        strategy_name=args.strategy,
        min_signals=args.min_signals,
        max_positions=args.max_positions,
        stock_limit=args.limit,
    )

    if result['status'] == 'success':
        print(f"\n{'='*50}")
        print(f"回测结果")
        print(f"{'='*50}")
        print(f"初始资金:     {result['initial_cash']:,.0f}")
        print(f"最终价值:     {result['final_value']:,.0f}")
        print(f"总收益率:     {result['total_return_pct']:.2f}%")
        print(f"年化收益率:   {result['annual_return_pct']:.2f}%")
        print(f"总交易次数:   {result['total_trades']}")
        print(f"回测股票数:   {result['stock_count']}")
        print(f"回测天数:     {result['days']}")
        print(f"{'='*50}")
    else:
        print(f"回测失败: {result['error']}")
