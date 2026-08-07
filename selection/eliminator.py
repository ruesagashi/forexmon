import random
import pandas as pd
import numpy as np
from typing import List, Tuple
from loguru import logger
from models.specialist import Specialist

class BacktestEngine:
    """
    Simulates trading over historical data to evaluate a Specialist.
    """
    
    @staticmethod
    def calculate_max_drawdown(trades: list) -> float:
        """
        Hitung Max Drawdown sebagai persentase dari peak equity.
        Return nilai 0.0 - 1.0 (bukan lebih dari 1.0).
        """
        if not trades:
            return 0.0
        
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        
        # Gunakan win/loss count bukan nilai PnL absolut
        # untuk menghindari distorsi dari harga pair yang tinggi
        for t in trades:
            if t["result"] == 1:  # WIN
                cumulative += 1.0
            else:  # LOSS
                cumulative -= 1.0
                
            if cumulative > peak:
                peak = cumulative
                
            if peak > 0:
                dd = (peak - cumulative) / peak
                if dd > max_dd:
                    max_dd = dd
                    
        return min(max_dd, 1.0)  # Cap di 100%

    def __init__(self, df: pd.DataFrame, specialist: Specialist):
        self.df = df
        self.specialist = specialist
        self.trades = []

    def run(self) -> dict:
        """
        Run the backtest and return a list of trades and performance metrics.
        """
        # Get features
        features = self.df[self.specialist.feature_cols]
        # Predict actions (0: HOLD, 1: BUY, 2: SELL)
        predictions = self.specialist.model.predict(features)
        
        opens = self.df['open'].values
        highs = self.df['high'].values
        lows = self.df['low'].values
        atrs = self.df['atr'].values
        
        from config.settings import settings
        sl_mult = settings.SL_ATR_MULT
        tp_mult = settings.TP_ATR_MULT
        horizon = 8
        
        n = len(self.df)
        for i in range(n - 1):
            pred = predictions[i]
            if pred == 0:
                continue
                
            entry_price = opens[i+1]
            atr = atrs[i]
            
            is_buy = (pred == 1)
            if is_buy:
                sl = entry_price - (atr * sl_mult)
                tp = entry_price + (atr * tp_mult)
            else:
                sl = entry_price + (atr * sl_mult)
                tp = entry_price - (atr * tp_mult)
                
            # Simulate forward to see if hit SL or TP first
            result = 0  # 0: hold/timeout, 1: win, -1: loss
            pnl = 0.0
            
            for j in range(i+1, min(i+1+horizon, n)):
                h = highs[j]
                l = lows[j]
                
                if is_buy:
                    if l <= sl and h >= tp:
                        # Ambiguous intra-candle, assume worst case
                        result = -1
                        pnl = - (atr * sl_mult)
                        break
                    elif l <= sl:
                        result = -1
                        pnl = - (atr * sl_mult)
                        break
                    elif h >= tp:
                        result = 1
                        pnl = (atr * tp_mult)
                        break
                else:
                    if h >= sl and l <= tp:
                        result = -1
                        pnl = - (atr * sl_mult)
                        break
                    elif h >= sl:
                        result = -1
                        pnl = - (atr * sl_mult)
                        break
                    elif l <= tp:
                        result = 1
                        pnl = (atr * tp_mult)
                        break
            
            if result != 0:
                self.trades.append({
                    "entry_idx": i+1,
                    "is_buy": is_buy,
                    "pnl": pnl,
                    "result": result
                })
                
        wins = sum(1 for t in self.trades if t["result"] == 1)
        losses = sum(1 for t in self.trades if t["result"] == -1)
        total_trades = wins + losses
        winrate = wins / total_trades if total_trades > 0 else 0.0
        
        gross_profit = sum(t["pnl"] for t in self.trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in self.trades if t["pnl"] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 99.0
        
        return {
            "total_trades": total_trades,
            "winrate": winrate,
            "profit_factor": profit_factor,
            "max_drawdown": self.calculate_max_drawdown(self.trades),
            "trades": self.trades
        }

class MonteCarloSimulator:
    """
    Performs Monte Carlo simulations on backtest trades.
    """
    def __init__(self, trades: List[dict], num_simulations: int = 1000):
        self.trades = trades
        self.num_simulations = num_simulations

    def run(self) -> float:
        """
        Runs simulations and returns probability of profitability.
        """
        if not self.trades:
            return 0.0
            
        pnls = [t["pnl"] for t in self.trades]
        profitable_runs = 0
        
        for _ in range(self.num_simulations):
            # Resample with replacement
            simulated_pnls = random.choices(pnls, k=len(pnls))
            total_pnl = sum(simulated_pnls)
            if total_pnl > 0:
                profitable_runs += 1
                
        return profitable_runs / self.num_simulations

def run_pre_filter(df: pd.DataFrame, specialist: Specialist, regime_confidence: float = 1.0) -> Tuple[bool, dict]:
    """
    Menjalankan Stage 1 Pre-Filter (Backtest & Monte Carlo).
    Mengembalikan (is_passed, metrics)
    """
    logger.info(f"[Pre-Filter] Memulai evaluasi untuk model {specialist.regime.value} (Confidence: {regime_confidence:.2f})")
    
    if regime_confidence < 0.70:
        logger.warning(f"[Pre-Filter] GAGAL: Regime confidence {regime_confidence:.2f} < 0.70")
        return False, {"reason": "Low regime confidence"}
        
    engine = BacktestEngine(df, specialist)
    results = engine.run()
    
    from config.settings import settings
    total_trades = results["total_trades"]
    
    if total_trades < settings.MIN_BACKTEST_TRADES:
        logger.warning(f"[Pre-Filter] GAGAL: Trade terlalu sedikit ({total_trades} < {settings.MIN_BACKTEST_TRADES})")
        return False, results
        
    if total_trades < 50:
        logger.warning(f"Specialist {specialist.id}: only {total_trades} trades in backtest")
        
    if results["winrate"] < 0.50 or results["profit_factor"] < 1.00:
        logger.warning(f"[Pre-Filter] GAGAL: WR={results['winrate']:.2f}, PF={results['profit_factor']:.2f}")
        return False, results
        
    if results["max_drawdown"] > 0.60:
        logger.warning(f"[Pre-Filter] GAGAL: Max Drawdown {results['max_drawdown']:.1%} > 60%")
        return False, results
        
    mc = MonteCarloSimulator(results["trades"], num_simulations=1000)
    prob_profit = mc.run()
    
    if prob_profit < 0.65:
        logger.warning(f"[Pre-Filter] GAGAL: Probabilitas profit MC {prob_profit:.2f} < 0.65")
        return False, results
        
    logger.success(f"[Pre-Filter] LULUS: WR={results['winrate']:.2f}, PF={results['profit_factor']:.2f}, MC_Prob={prob_profit:.2f}")
    
    # Update specialist initial metrics
    specialist.update_metrics(
        win_rate=results["winrate"],
        profit_factor=results["profit_factor"],
        trades_count=total_trades,
        recent_wr=results["winrate"]
    )
    
    return True, results
