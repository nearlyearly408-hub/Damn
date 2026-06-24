"""
Bot Scalping v20.2 — REVERSED LOGIC (CONTRARIAN) + STRICT RISK
====================================================
- Entry direction inverted (LONG <-> SHORT)
- Strict Take Profit 0.5% & Stop Loss 0.5% (No Trailing / ATR bullshit)
- Ultra-fast real-time monitoring anti-jebol
"""

import os
import time
import math
import threading
import queue
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from binance.client import Client
import ta

load_dotenv()
client = Client(os.getenv("API_KEY"), os.getenv("API_SECRET"))
client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

LEVERAGE = 20
ORDER_USDT = 2.0
MAX_POSITIONS = 3

# Risk Management (Strict Fixed Percentage)
FIXED_SL_PCT = 0.005  # 0.5% Limit
FIXED_TP_PCT = 0.005  # 0.5% Limit

# ═══════════════════════════════════════════════════════════════════════════
#  SYMBOLS
# ═══════════════════════════════════════════════════════════════════════════
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "TRXUSDT", "DOTUSDT",
    "LINKUSDT", "MATICUSDT", "LTCUSDT", "ATOMUSDT", "UNIUSDT",
    "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT",
    "SUIUSDT", "SEIUSDT", "FETUSDT", "WLDUSDT", "AAVEUSDT",
    "ORDIUSDT", "TONUSDT", "1000PEPEUSDT", "WIFUSDT", "JUPUSDT",
    "FTMUSDT", "SANDUSDT", "MANAUSDT", "GALAUSDT", "APEUSDT",
    "CRVUSDT", "1000SHIBUSDT", "COMPUSDT", "MKRUSDT", "SNXUSDT",
]
SYMBOLS = list(dict.fromkeys(SYMBOLS))

# Scanning & Monitoring
SCAN_INTERVAL = 2.0
MONITOR_INT = 0.2  # Dipercepat jadi 0.2 detik biar real-time dan nggak jebol
BATCH_SIZE = 15
MAX_WORKERS = 5
SLOT_FILL_INT = 0.01

# Scoring & Filter
MIN_SCORE = 55
SLIPPAGE_GUARD = 0.0015
TTL_5M = 2

# Kill Switch
DAILY_LOSS = -20.0
CONSEC_MAX = 15
CONSEC_PAUSE = 10

# Learning
LEARNING_WINDOW = 200
MIN_TRADES_FOR_WEIGHT = 20

# ═══════════════════════════════════════════════════════════════════════════
#  MARKET REGIME DETECTION
# ═══════════════════════════════════════════════════════════════════════════

class MarketRegime:
    REGIME_TRENDING_BULL = "TRENDING_BULL"
    REGIME_TRENDING_BEAR = "TRENDING_BEAR"
    REGIME_RANGE = "RANGE"
    REGIME_VOLATILE = "VOLATILE"
    REGIME_EXHAUSTION = "EXHAUSTION"
    
    @staticmethod
    def detect(df: pd.DataFrame) -> Tuple[str, float, float]:
        if df is None or len(df) < 55:
            return MarketRegime.REGIME_RANGE, 0, 0
        row = df.iloc[-2]
        prev = df.iloc[-3]
        close = row["close"]
        e5, e9, e21, e50 = row["e5"], row["e9"], row["e21"], row["e50"]
        atr = row["atr"]
        atr_prev = prev["atr"]
        adx = row["adx"]
        bull_stack = close > e5 > e9 > e21 > e50
        bear_stack = close < e5 < e9 < e21 < e50
        mild_bull = close > e9 > e21
        mild_bear = close < e9 < e21
        strong_trend = adx > 25
        very_strong_trend = adx > 35
        atr_expand = (atr / atr_prev) > 1.2 if atr_prev > 0 else False
        atr_collapse = (atr / atr_prev) < 0.8 if atr_prev > 0 else False
        m5 = row["m5"]
        m5_prev = prev["m5"]
        decelerating = (abs(m5) < abs(m5_prev)) if not np.isnan(m5_prev) else False
        
        if very_strong_trend and bull_stack:
            return MarketRegime.REGIME_TRENDING_BULL, min(adx, 100), 1.0
        elif very_strong_trend and bear_stack:
            return MarketRegime.REGIME_TRENDING_BEAR, min(adx, 100), -1.0
        elif strong_trend and (bull_stack or mild_bull):
            return MarketRegime.REGIME_TRENDING_BULL, min(adx, 80), 0.7
        elif strong_trend and (bear_stack or mild_bear):
            return MarketRegime.REGIME_TRENDING_BEAR, min(adx, 80), -0.7
        elif atr_expand and adx < 20:
            return MarketRegime.REGIME_VOLATILE, 50, 0
        elif (atr_collapse and decelerating) or (adx > 20 and adx < 35 and decelerating):
            return MarketRegime.REGIME_EXHAUSTION, 40, 1 if m5 > 0 else -1
        else:
            return MarketRegime.REGIME_RANGE, 30, 0


# ═══════════════════════════════════════════════════════════════════════════
#  SELF-LEARNING SIGNAL WEIGHTING
# ═══════════════════════════════════════════════════════════════════════════

class SignalWeights:
    def __init__(self):
        self.weights = {
            "ema_bull_stack": 35, "ema_mild_bull": 26, "ema_weak_bull": 14,
            "mom_strong": 30, "mom_moderate": 20,
            "macd_cross_up": 22, "macd_strengthen": 15,
            "orderflow_buy_climax": 25, "orderflow_buy_high": 14,
            "rsi_extreme_ob": 25, "rsi_high": 12,
            "ema_bear_stack": 35, "ema_mild_bear": 26, "ema_weak_bear": 14,
            "mom_strong_neg": 30, "mom_moderate_neg": 20,
            "macd_cross_down": 22, "macd_strengthen_neg": 15,
            "orderflow_sell_climax": 25, "orderflow_sell_high": 14,
            "rsi_extreme_os": 25, "rsi_low": 12,
        }
        self.history = defaultdict(list)
        self.adaptive_enabled = True
    
    def record_outcome(self, signals: List[str], won: bool):
        for sig in signals:
            base_sig = sig.split('[')[0].strip()
            if base_sig in self.weights:
                self.history[base_sig].append(1 if won else 0)
                if len(self.history[base_sig]) > LEARNING_WINDOW:
                    self.history[base_sig] = self.history[base_sig][-LEARNING_WINDOW:]
    
    def get_adjusted_weight(self, signal_name: str) -> float:
        if not self.adaptive_enabled:
            return self.weights.get(signal_name, 10)
        base = signal_name.split('[')[0].strip()
        hist = self.history.get(base, [])
        if len(hist) < MIN_TRADES_FOR_WEIGHT:
            return self.weights.get(base, 10)
        win_rate = sum(hist) / len(hist)
        factor = max(0.5, min(1.5, 0.5 + win_rate))
        return self.weights.get(base, 10) * factor


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL SCORING
# ═══════════════════════════════════════════════════════════════════════════

class SignalScorer:
    def __init__(self, signal_weights: SignalWeights):
        self.weights = signal_weights
    
    def get_signal(self, df: pd.DataFrame, symbol: str = None) -> Tuple[Optional[str], int, List[str], float, str, float]:
        if df is None or len(df) < 55:
            return None, 0, [], 0.0, "UNKNOWN", 0.0
        
        regime, strength, bias = MarketRegime.detect(df)
        long_score, long_signals = self._score_long(df)
        short_score, short_signals = self._score_short(df)
        
        atr = df["atr"].iloc[-2]
        
        # Sederhanakan filter, langsung berikan arah murni berdasarkan skor
        if long_score >= MIN_SCORE and long_score > short_score:
            return "LONG", long_score, long_signals, atr, regime, bias
        elif short_score >= MIN_SCORE and short_score > long_score:
            return "SHORT", short_score, short_signals, atr, regime, bias
            
        return None, max(long_score, short_score), [], atr, regime, bias
    
    def _score_long(self, df: pd.DataFrame) -> Tuple[int, List[str]]:
        row, prev, prev2 = df.iloc[-2], df.iloc[-3], df.iloc[-4]
        score = 0
        signals = []
        p, e5, e9, e21, e50 = row["close"], row["e5"], row["e9"], row["e21"], row["e50"]
        
        if p < e5 < e9 < e21 < e50:
            w = self.weights.get_adjusted_weight("ema_bear_stack")
            score += w; signals.append(f"EMA5↓[{w:.0f}]")
        
        m5 = row["m5"]
        if m5 < -0.003:
            w = self.weights.get_adjusted_weight("mom_strong_neg")
            score += w; signals.append(f"Mom{m5*100:.1f}%↓[{w:.0f}]")
            
        mh, mh_p, mh_p2 = row["mh"], prev["mh"], prev2["mh"]
        if mh_p >= 0 and mh < 0:
            w = self.weights.get_adjusted_weight("macd_cross_down")
            score += w; signals.append(f"MACD_X↓[{w:.0f}]")
            
        rsi = row["rsi"]
        if rsi < 32:
            w = self.weights.get_adjusted_weight("rsi_extreme_os")
            score += w; signals.append(f"RSI{rsi:.0f}OS[{w:.0f}]")
            
        return score, signals
    
    def _score_short(self, df: pd.DataFrame) -> Tuple[int, List[str]]:
        row, prev, prev2 = df.iloc[-2], df.iloc[-3], df.iloc[-4]
        score = 0
        signals = []
        p, e5, e9, e21, e50 = row["close"], row["e5"], row["e9"], row["e21"], row["e50"]
        
        if p > e5 > e9 > e21 > e50:
            w = self.weights.get_adjusted_weight("ema_bull_stack")
            score += w; signals.append(f"EMA5↑[{w:.0f}]")
            
        m5 = row["m5"]
        if m5 > 0.003:
            w = self.weights.get_adjusted_weight("mom_strong")
            score += w; signals.append(f"Mom+{m5*100:.1f}%↑[{w:.0f}]")
            
        mh, mh_p, mh_p2 = row["mh"], prev["mh"], prev2["mh"]
        if mh_p <= 0 and mh > 0:
            w = self.weights.get_adjusted_weight("macd_cross_up")
            score += w; signals.append(f"MACD_X↑[{w:.0f}]")
            
        rsi = row["rsi"]
        if rsi > 68:
            w = self.weights.get_adjusted_weight("rsi_extreme_ob")
            score += w; signals.append(f"RSI{rsi:.0f}OB[{w:.0f}]")
            
        return score, signals


# ═══════════════════════════════════════════════════════════════════════════
#  RISK MANAGER (STRICT FIXED PERCENTAGE)
# ═══════════════════════════════════════════════════════════════════════════

class RiskManager:
    @staticmethod
    def calculate_sl_tp(entry_price: float, direction: str) -> Tuple[float, float, float, float]:
        # Dibuat FIX 0.5% dari harga entry. Tidak peduli ATR atau volatilitas.
        sl_distance = entry_price * FIXED_SL_PCT
        tp_distance = entry_price * FIXED_TP_PCT
        
        if direction == "LONG":
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance
            
        return sl_price, tp_price, FIXED_SL_PCT, FIXED_TP_PCT

# ═══════════════════════════════════════════════════════════════════════════
#  TRADE RECORDER & LEARNING LAYER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    won: bool
    regime: str
    signals: List[str]
    score: float
    hold_seconds: float
    timestamp: float = field(default_factory=time.time)


class LearningLayer:
    def __init__(self, signal_weights: SignalWeights):
        self.signal_weights = signal_weights
        self.trades: List[TradeRecord] = []
        self.stats_by_regime = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})
    
    def add_trade(self, trade: TradeRecord):
        self.trades.append(trade)
        regime = trade.regime
        self.stats_by_regime[regime]["wins"] += 1 if trade.won else 0
        self.stats_by_regime[regime]["losses"] += 0 if trade.won else 1
        self.stats_by_regime[regime]["pnl"] += trade.pnl
        self.signal_weights.record_outcome(trade.signals, trade.won)
        if len(self.trades) > 1000:
            self.trades = self.trades[-500:]
    
    def get_winrate_by_regime(self, regime: str) -> float:
        stats = self.stats_by_regime[regime]
        total = stats["wins"] + stats["losses"]
        return stats["wins"] / total if total > 0 else 0.5
    
    def get_global_winrate(self) -> float:
        total_wins = sum(s["wins"] for s in self.stats_by_regime.values())
        total_losses = sum(s["losses"] for s in self.stats_by_regime.values())
        total = total_wins + total_losses
        return total_wins / total if total > 0 else 0.5


# ═══════════════════════════════════════════════════════════════════════════
#  BOT STATE & UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

_precision_cache = {}
_ohlcv_cache = {}
_ticker_cache = {}
_ticker_ts = 0
_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
_rescan_q = queue.Queue()
_hot_syms = deque(maxlen=30)

_macro = {"btc": "UNKNOWN"}
_ks = {"active": False, "reason": "", "resume": 0, "consec": 0, "daily": 0.0, "day_reset": 0}
_stats = {
    "trades": 0, "wins": 0, "losses": 0, "pnl": 0.0, "best": 0.0, "worst": 0.0,
    "extreme_tp": 0, "hard_sl": 0,
    "hist": deque(maxlen=200), "start": time.time(),
}

def get_precision(symbol):
    if symbol in _precision_cache: return _precision_cache[symbol]
    try:
        info = client.futures_exchange_info()
        for s in info['symbols']:
            if s['symbol'] == symbol:
                prec = int(s['quantityPrecision'])
                _precision_cache[symbol] = prec
                return prec
    except: pass
    return 2

def qty(symbol, price):
    raw_qty = (ORDER_USDT * LEVERAGE) / price
    prec = get_precision(symbol)
    return round(raw_qty, prec)

def price_live(symbol):
    try: return float(client.futures_symbol_ticker(symbol=symbol)["price"])
    except: return 0.0

def get_all_prices():
    # Menggunakan endpoint weight yang sangat ringan (Weight: 2) untuk scan instan!
    try:
        raw = client.futures_symbol_ticker()
        return {t["symbol"]: float(t["price"]) for t in raw}
    except:
        return {}

def tickers_all():
    global _ticker_cache, _ticker_ts
    now = time.time()
    # Cache hanya 0.5 detik untuk ngeringanin API, gak selama 2 detik lagi!
    if now - _ticker_ts < 0.5 and _ticker_cache: return _ticker_cache
    try:
        raw = client.futures_ticker()
        _ticker_cache = {
            t["symbol"]: {
                "pct": float(t["priceChangePercent"]),
                "last": float(t["lastPrice"])
            } for t in raw
        }
        _ticker_ts = now
        return _ticker_cache
    except: return _ticker_cache

def ohlcv(symbol, interval, limit=100):
    key, now = (symbol, interval), time.time()
    ttl = TTL_5M
    if key in _ohlcv_cache and now - _ohlcv_cache[key][0] < ttl:
        return _ohlcv_cache[key][1]
    try:
        kl = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(kl, columns=["time","open","high","low","close","volume",
                                        "ct","qv","trades","tbbase","tbquote","ignore"])
        for c in ["open","high","low","close","volume","tbbase","tbquote"]:
            df[c] = df[c].astype(float)
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
        df["mh"] = ta.trend.MACD(df["close"], 12, 26, 9).macd_diff()
        df["e5"] = ta.trend.EMAIndicator(df["close"], 5).ema_indicator()
        df["e9"] = ta.trend.EMAIndicator(df["close"], 9).ema_indicator()
        df["e21"] = ta.trend.EMAIndicator(df["close"], 21).ema_indicator()
        df["e50"] = ta.trend.EMAIndicator(df["close"], 50).ema_indicator()
        df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range()
        df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], 14).adx()
        df["m5"] = (df["close"] - df["close"].shift(5)) / df["close"].shift(5)
        _ohlcv_cache[key] = (now, df)
        return df
    except:
        return _ohlcv_cache.get(key, (None, None))[1]

def ks_check():
    k, now = _ks, time.time()
    if k["active"] and now >= k["resume"]:
        k["active"] = False; k["consec"] = 0
    if k["active"]: return True, k["reason"]
    day = now - (now % 86400)
    if day > k["day_reset"]: k["daily"] = 0.0; k["day_reset"] = day
    if k["daily"] <= DAILY_LOSS:
        k["active"] = True
        k["reason"] = f"daily({k['daily']:.2f})"
        k["resume"] = day + 86400
        return True, k["reason"]
    if k["consec"] >= CONSEC_MAX:
        k["active"] = True
        k["reason"] = f"consec({k['consec']})"
        k["resume"] = now + CONSEC_PAUSE
        return True, k["reason"]
    return False, ""

def ks_upd(pnl):
    _ks["daily"] += pnl
    _ks["consec"] = 0 if pnl >= 0 else _ks["consec"] + 1

# ═══════════════════════════════════════════════════════════════════════════
#  TRADING STATE
# ═══════════════════════════════════════════════════════════════════════════

live_positions = {}
trade_log = []
signal_weights = SignalWeights()
scorer = SignalScorer(signal_weights)
learning = LearningLayer(signal_weights)

# ═══════════════════════════════════════════════════════════════════════════
#  CORE TRADING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def live_open(orig_direction, score, sigs, price, regime, bias, sym):
    with _lock:
        if sym in live_positions or len(live_positions) >= MAX_POSITIONS:
            return
        live_positions[sym] = {"_r": True}
    
    px_now = price_live(sym)
    if px_now > 0:
        slip = abs(px_now - price) / price
        if slip > SLIPPAGE_GUARD:
            with _lock: live_positions.pop(sym, None)
            return
        price = px_now
    
    try:
        q_val = qty(sym, price)
    except:
        with _lock: live_positions.pop(sym, None)
        return
    
    # REVERSED LOGIC DI SINI
    actual_side = "SHORT" if orig_direction == "LONG" else "LONG"
    
    # Hitung SL & TP strictly dari 0.5% (Tidak peduli kebalikan atau nggak, math-nya sama)
    sl_price, tp_price, sl_pct, tp_pct = RiskManager.calculate_sl_tp(price, actual_side)
    
    pos = {
        "side": actual_side,
        "entry": price,
        "qty": q_val,
        "open_time": time.time(),
        "score": score,
        "sigs": sigs,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "sl_pct": sl_pct,
        "tp_pct": tp_pct,
        "regime": regime,
        "bias": bias,
        "orig_direction": orig_direction
    }
    with _lock: live_positions[sym] = pos
    
    d = "🟢" if actual_side == "LONG" else "🔴"
    print(f"\n  {d} [REVERSED] {sym} {actual_side} (orig {orig_direction}) @{price:.6g} | SL:{sl_pct*100:.1f}% TP:{tp_pct*100:.1f}%")
    print(f"        Original signals: {' | '.join(sigs[:5])}")
    _stats["trades"] += 1

def live_close(sym, reason, price=None):
    with _lock:
        pos = live_positions.pop(sym, None)
    if pos is None or pos.get("_r"): return
    
    if price is None: price = price_live(sym)
    if price == 0: return
    
    side, entry, q_val = pos["side"], pos["entry"], pos["qty"]
    gross_pnl = (price - entry) * q_val if side == "LONG" else (entry - price) * q_val
    fee_rate = 0.0005  # Taker fee
    total_fee = (entry * q_val + price * q_val) * fee_rate
    pnl = gross_pnl - total_fee
    pct = (price - entry) / entry * 100 if side == "LONG" else (entry - price) / entry * 100
    hold = time.time() - pos["open_time"]
    won = pnl >= 0
    e = "🟢" if won else "🔴"
    
    print(f"  {e} [REVERSED] {sym} {side} CLOSE — {reason}")
    print(f"     {entry:.6g}→{price:.6g} ({pct:+.3f}%) hold:{hold:.0f}s | PnL:{pnl:+.5f}U")
    
    trade = TradeRecord(
        symbol=sym, direction=side, entry_price=entry, exit_price=price,
        pnl=pnl, won=won, regime=pos.get("regime", "UNKNOWN"),
        signals=pos.get("sigs", []), score=pos.get("score", 0),
        hold_seconds=hold
    )
    learning.add_trade(trade)
    
    _stats["pnl"] += pnl
    _stats["hist"].append(pnl)
    ks_upd(pnl)
    if won:
        _stats["wins"] += 1
        if pnl > _stats["best"]: _stats["best"] = pnl
    else:
        _stats["losses"] += 1
        if pnl < _stats["worst"]: _stats["worst"] = pnl
    
    if "TP" in reason: _stats["extreme_tp"] += 1
    elif "SL" in reason: _stats["hard_sl"] += 1
    
    trade_log.append({
        "sym": sym, "side": side, "entry": round(entry, 7), "exit": round(price, 7),
        "pnl": round(pnl, 5), "reason": reason, "hold": int(hold),
    })
    _hot_syms.appendleft(sym)
    _rescan_q.put(1)
    print_inline()

def monitor_positions():
    # Tarik semua harga dalam 1x request super enteng (weight: 2) -> eksekusi gak akan telat!
    prices = get_all_prices()
    if not prices: return
    
    for sym in list(live_positions.keys()):
        pos = live_positions.get(sym)
        if pos is None or pos.get("_r"): continue
        
        px = prices.get(sym, 0.0)
        if px == 0: continue
        
        side = pos["side"]
        sl_px = pos["sl_price"]
        tp_px = pos["tp_price"]
        
        if side == "LONG":
            if px <= sl_px:
                live_close(sym, "HardSL", px); continue
            if px >= tp_px:
                live_close(sym, "ExtremeTP", px); continue
        else:
            if px >= sl_px:
                live_close(sym, "HardSL", px); continue
            if px <= tp_px:
                live_close(sym, "ExtremeTP", px); continue

# ═══════════════════════════════════════════════════════════════════════════
#  SCANNER THREAD
# ═══════════════════════════════════════════════════════════════════════════

def scan_one(sym):
    try:
        time.sleep(0.002)
        df = ohlcv(sym, Client.KLINE_INTERVAL_5MINUTE, 100)
        if df is None: return None
        
        px = df["close"].iloc[-2]
        if px == 0: return None
        
        direction, score, sigs, atr_val, regime, bias = scorer.get_signal(df, sym)
        if direction is None: return None
        
        px_live = price_live(sym)
        if px_live == 0: return None
        
        return (sym, direction, score, sigs, px_live, regime, bias)
    except Exception as e:
        return None

def scan_batch(syms):
    res = []
    fut = {_executor.submit(scan_one, s): s for s in syms[:BATCH_SIZE]}
    for f in as_completed(fut, timeout=5):
        try:
            if r := f.result(timeout=1): res.append(r)
        except: pass
    return res

def top_movers(syms, n=30):
    tk, ss = tickers_all(), set(syms)
    mv = [(s, abs(d["pct"])) for s, d in tk.items() if s in ss]
    return [s for s, _ in sorted(mv, key=lambda x: x[1], reverse=True)[:n]]

# ═══════════════════════════════════════════════════════════════════════════
#  PRINTING
# ═══════════════════════════════════════════════════════════════════════════

def print_inline():
    n = _stats["wins"] + _stats["losses"]
    wr = _stats["wins"] / n * 100 if n else 0
    pnl, e = _stats["pnl"], "💚" if _stats["pnl"] >= 0 else "🔴"
    print(f"       ┌ [v20.2 FIX] {n}T WR:{wr:.0f}% W:{_stats['wins']} L:{_stats['losses']} {e}PnL:{pnl:+.4f}U")
    print(f"       └ TP:{_stats['extreme_tp']} SL:{_stats['hard_sl']} | Regime WR: {learning.get_winrate_by_regime('TRENDING_BULL'):.0%}")

def print_full():
    n = _stats["wins"] + _stats["losses"]
    wr = _stats["wins"] / n * 100 if n else 0
    pnl = _stats["pnl"]
    sess = (time.time() - _stats["start"]) / 3600
    tph = n / sess if sess > 0 else 0
    e = "💚" if pnl >= 0 else "🔴"
    print(f"\n  {'─'*70}")
    print(f"    ✅ REVERSED LOGIC v20.2 — STRICT 0.5% SL/TP")
    print(f"    🎯 {n}T WR:{wr:.0f}% W:{_stats['wins']} L:{_stats['losses']} ({tph:.1f}T/hr)")
    print(f"    {e} PnL Net:{pnl:+.5f}U Best:{_stats['best']:+.5f} Worst:{_stats['worst']:+.5f}")
    print(f"    💰 TP:{_stats['extreme_tp']} SL:{_stats['hard_sl']}")
    print(f"    📊 Learning: Global WR {learning.get_global_winrate():.1%}")
    print(f"    ⚙️  Logic Reversed | 0.5% Strict Mode | Fast Polling Active")
    if trade_log:
        print(f"    📋 Last 5:")
        for t in trade_log[-5:]:
            em = "🟢" if t["pnl"] > 0 else "🔴"
            print(f"       {em} {t['sym']:<16} {t['side']} {t['pnl']:+.5f}U {t['hold']}s — {t['reason']}")
    print(f"  {'─'*70}")

# ═══════════════════════════════════════════════════════════════════════════
#  THREADS
# ═══════════════════════════════════════════════════════════════════════════

def t_monitor():
    while True:
        try:
            if live_positions:
                monitor_positions()
        except: pass
        time.sleep(MONITOR_INT) # Jauh lebih cepat (0.2s)

def t_slot_filler(syms):
    scan_idx = 0
    n_bat = max(1, math.ceil(len(syms) / BATCH_SIZE))
    while True:
        try:
            slots = MAX_POSITIONS - len(live_positions)
            if slots <= 0 or ks_check()[0]:
                time.sleep(SLOT_FILL_INT)
                continue
            hot = [s for s in _hot_syms if s not in live_positions]
            mv = top_movers(syms, 30)
            mv = [s for s in mv if s not in live_positions]
            bs = scan_idx * BATCH_SIZE
            reg = [s for s in syms[bs:bs+BATCH_SIZE] if s not in live_positions and s not in mv]
            scan_idx = (scan_idx + 1) % n_bat
            scan_list = list(dict.fromkeys(hot[:5] + mv[:20] + reg[:15]))[:BATCH_SIZE]
            if not scan_list:
                time.sleep(SLOT_FILL_INT)
                continue
            res = scan_batch(scan_list)
            if res:
                res.sort(key=lambda x: x[2], reverse=True)
                for r in res[:slots]:
                    if len(live_positions) >= MAX_POSITIONS: break
                    sym, orig_dir, sc, sg, px, regime, bias = r
                    live_open(orig_dir, sc, sg, px, regime, bias, sym)
        except: pass
        time.sleep(SLOT_FILL_INT)

def t_rescan(syms):
    while True:
        try:
            _rescan_q.get(timeout=5)
            time.sleep(0.05)
            slots = MAX_POSITIONS - len(live_positions)
            if slots <= 0 or ks_check()[0]: continue
            hot = [s for s in _hot_syms if s not in live_positions]
            rest = [s for s in syms if s not in live_positions and s not in hot]
            res = scan_batch((hot + rest)[:30])
            if res:
                res.sort(key=lambda x: x[2], reverse=True)
                for r in res[:slots]:
                    if len(live_positions) >= MAX_POSITIONS: break
                    sym, orig_dir, sc, sg, px, regime, bias = r
                    live_open(orig_dir, sc, sg, px, regime, bias, sym)
        except: pass

def t_macro():
    while True:
        try:
            df_btc = ohlcv("BTCUSDT", Client.KLINE_INTERVAL_5MINUTE, 80)
            if df_btc is not None:
                regime, _, _ = MarketRegime.detect(df_btc)
                _macro["btc"] = regime
        except: pass
        time.sleep(10)

# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def run_bot():
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  ✅ REVERSED LOGIC v20.2 — STRICT TP & SL FIX                      ║")
    print("║  ✅ Entry dibalik (LONG<->SHORT)                                   ║")
    print("║  ✅ STRICT SL/TP 0.5% (Total Nett: TP 0.4% & SL 0.6% pasca fee)    ║")
    print("║  ✅ Real-time Polling (Anti-Jebol System Active)                   ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    try:
        valid = {s["symbol"] for s in client.futures_exchange_info()["symbols"] if s["status"] == "TRADING"}
        syms = list(dict.fromkeys([s for s in SYMBOLS if s in valid]))
    except:
        syms = list(dict.fromkeys(SYMBOLS))
    print(f"  📋 {len(syms)} simbol aktif terpantau")
    threading.Thread(target=t_monitor, daemon=True).start()
    threading.Thread(target=t_slot_filler, args=(syms,), daemon=True).start()
    threading.Thread(target=t_rescan, args=(syms,), daemon=True).start()
    threading.Thread(target=t_macro, daemon=True).start()
    time.sleep(2)
    tickers_all()
    cycle = 0
    while True:
        cycle += 1
        slots = MAX_POSITIONS - len(live_positions)
        print(f"\n{'═'*62}")
        print(f"  #{cycle} {time.strftime('%H:%M:%S')} BTC Regime:{_macro['btc']} ({len(live_positions)}/{MAX_POSITIONS}) PnL:{_stats['pnl']:+.4f}U")
        if (k := ks_check())[0]:
            print(f"  🚨 KS:{k[1]}")
        elif slots == 0:
            print(f"  ✅ Slots full")
        else:
            print(f"  🔍 {slots} slot kosong — Adaptive scanning (REVERSED)...")
        if cycle % 30 == 0:
            print_full()
        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    run_bot()
