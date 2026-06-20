"""
Bot Scalping v20.1.2 — REVERSED LOGIC (LOSS → PROFIT) - FIXED
=================================================================
BUG FIX dari versi sebelumnya:
1. TP & SL tidak lagi ditukar, melainkan dihitung ulang sesuai arah entry.
2. Kondisi close TP/SL di monitor_positions sudah benar.
3. Label output sudah sesuai (TP / SL).
"""

import os, time, math, threading, queue, json
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from dotenv import load_dotenv
from binance.client import Client
import ta

load_dotenv()
client = Client(os.getenv("API_KEY"), os.getenv("API_SECRET"))
client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

LEVERAGE      = 20
ORDER_USDT    = 2.0
MAX_POSITIONS = 3

HARD_SL_PCT    = 0.005    # 0.50% → di mode reversed menjadi TP kecil
EXTREME_TP_PCT = 0.020    # 2.00% → di mode reversed menjadi SL besar
FEE_RATE       = 0.001    # 0.10% per leg

REVERSED_MODE = True       # Aktifkan logika terbalik

# Trailing hanya digunakan jika REVERSED_MODE = False
TRAIL_PHASES = [
    (0.010, 0.005),
    (0.006, 0.004),
    (0.0045, 0.0035),
    (0.004, 0.003),
]
MIN_PROFIT_LOCK = 0.003

# ═══════════════════════════════════════════════════════════════════════════
#  SYMBOLS
# ═══════════════════════════════════════════════════════════════════════════

SYMBOLS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
    "ADAUSDT","DOGEUSDT","AVAXUSDT","TRXUSDT","DOTUSDT",
    "LINKUSDT","MATICUSDT","LTCUSDT","ATOMUSDT","UNIUSDT",
    "NEARUSDT","APTUSDT","ARBUSDT","OPUSDT","INJUSDT",
    "SUIUSDT","SEIUSDT","FETUSDT","WLDUSDT","AAVEUSDT",
    "ORDIUSDT","TONUSDT","1000PEPEUSDT","WIFUSDT","JUPUSDT",
    "FTMUSDT","SANDUSDT","MANAUSDT","GALAUSDT","APEUSDT",
    "CRVUSDT","1000SHIBUSDT","COMPUSDT","MKRUSDT","SNXUSDT",
]
SYMBOLS = list(dict.fromkeys(SYMBOLS))

# ── Scanning ──────────────────────────────────────────────────────────────
SCAN_INTERVAL = 2.0
MONITOR_INT   = 0.5
BATCH_SIZE    = 15
MAX_WORKERS   = 5
SLOT_FILL_INT = 0.01

# ── Scoring & Filter ──────────────────────────────────────────────────────
MIN_SCORE       = 55
MIN_SCORE_RANGE = 60
MIN_GAP         = 10
SLIPPAGE_GUARD  = 0.0015
TTL_5M          = 2

# ── Kill Switch ───────────────────────────────────────────────────────────
DAILY_LOSS   = -8.0
CONSEC_MAX   = 12
CONSEC_PAUSE = 15

# ── Learning ──────────────────────────────────────────────────────────────
LEARNING_WINDOW       = 200
MIN_TRADES_FOR_WEIGHT = 20

# ═══════════════════════════════════════════════════════════════════════════
#  MARKET REGIME DETECTION (TIDAK BERUBAH)
# ═══════════════════════════════════════════════════════════════════════════

class MarketRegime:
    REGIME_TRENDING_BULL = "TRENDING_BULL"
    REGIME_TRENDING_BEAR = "TRENDING_BEAR"
    REGIME_RANGE         = "RANGE"
    REGIME_VOLATILE      = "VOLATILE"
    REGIME_EXHAUSTION    = "EXHAUSTION"

    @staticmethod
    def detect(df: pd.DataFrame) -> Tuple[str, float, float]:
        if df is None or len(df) < 55:
            return MarketRegime.REGIME_RANGE, 0, 0
        row  = df.iloc[-2]
        prev = df.iloc[-3]
        close = row["close"]
        e5, e9, e21, e50 = row["e5"], row["e9"], row["e21"], row["e50"]
        atr, atr_prev = row["atr"], prev["atr"]
        adx = row["adx"]
        bull_stack = close > e5 > e9 > e21 > e50
        bear_stack = close < e5 < e9 < e21 < e50
        mild_bull  = close > e9 > e21
        mild_bear  = close < e9 < e21
        strong_trend = adx > 25
        very_strong_trend = adx > 35
        atr_expand = (atr / atr_prev) > 1.2 if atr_prev > 0 else False
        atr_collapse = (atr / atr_prev) < 0.8 if atr_prev > 0 else False
        m5 = row["m5"]
        m5_prev = prev["m5"]
        decelerating = (abs(m5) < abs(m5_prev)) if not np.isnan(m5_prev) else False

        if very_strong_trend and bull_stack:
            return MarketRegime.REGIME_TRENDING_BULL, min(adx,100), 1.0
        elif very_strong_trend and bear_stack:
            return MarketRegime.REGIME_TRENDING_BEAR, min(adx,100), -1.0
        elif strong_trend and (bull_stack or mild_bull):
            return MarketRegime.REGIME_TRENDING_BULL, min(adx,80), 0.7
        elif strong_trend and (bear_stack or mild_bear):
            return MarketRegime.REGIME_TRENDING_BEAR, min(adx,80), -0.7
        elif atr_expand and adx < 20:
            return MarketRegime.REGIME_VOLATILE, 50, 0
        elif (atr_collapse and decelerating) or (20 < adx < 35 and decelerating):
            bias = 1 if m5 > 0 else -1
            return MarketRegime.REGIME_EXHAUSTION, 40, bias
        else:
            return MarketRegime.REGIME_RANGE, 30, 0

# ═══════════════════════════════════════════════════════════════════════════
#  EXHAUSTION CONFIRMATION (TIDAK BERUBAH)
# ═══════════════════════════════════════════════════════════════════════════

class ExhaustionConfirmation:
    @staticmethod
    def check_short_exhaustion(df): pass  # ... sama seperti sebelumnya
    @staticmethod
    def check_long_exhaustion(df): pass   # ... sama seperti sebelumnya

# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL WEIGHTS, SCORER (TIDAK BERUBAH)
# ═══════════════════════════════════════════════════════════════════════════

class SignalWeights:
    def __init__(self): pass  # ... sama
class SignalScorer:
    def __init__(self, signal_weights): pass  # ... sama

# ═══════════════════════════════════════════════════════════════════════════
#  RISK MANAGER – DIPERBAIKI UNTUK REVERSED
# ═══════════════════════════════════════════════════════════════════════════

class RiskManager:
    @staticmethod
    def calculate_reversed_tpsl(entry_price: float, direction: str) -> Tuple[float, float]:
        """
        Menghitung TP & SL untuk mode reversed.
        TP = 0.50% ke arah profit, SL = 2.00% ke arah loss.
        """
        if direction == "LONG":
            tp_price = entry_price * (1 + HARD_SL_PCT)      # profit jika naik 0.50%
            sl_price = entry_price * (1 - EXTREME_TP_PCT)   # loss jika turun 2.00%
        else:  # SHORT
            tp_price = entry_price * (1 - HARD_SL_PCT)      # profit jika turun 0.50%
            sl_price = entry_price * (1 + EXTREME_TP_PCT)   # loss jika naik 2.00%
        return tp_price, sl_price

    @staticmethod
    def calculate_sl_tp(entry_price: float, direction: str) -> Tuple[float, float, float, float]:
        sl_pct = HARD_SL_PCT
        tp_pct = EXTREME_TP_PCT
        if direction == "LONG":
            sl_price = entry_price * (1 - sl_pct)
            tp_price = entry_price * (1 + tp_pct)
        else:
            sl_price = entry_price * (1 + sl_pct)
            tp_price = entry_price * (1 - tp_pct)
        return sl_price, tp_price, sl_pct, tp_pct

    @staticmethod
    def get_trail_stop_price(entry_price, direction, best_price, best_gross_pct):
        # ... tidak berubah, hanya dipakai jika tidak reversed
        pass

# ═══════════════════════════════════════════════════════════════════════════
#  TRADE RECORDER, LEARNING (TIDAK BERUBAH)
# ═══════════════════════════════════════════════════════════════════════════

# ... (persis seperti kode sebelumnya)

# ═══════════════════════════════════════════════════════════════════════════
#  BOT STATE (TIDAK BERUBAH)
# ═══════════════════════════════════════════════════════════════════════════

_precision_cache = {}
_ohlcv_cache = {}
_ticker_cache = {}
_ticker_ts = 0.0
_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
_rescan_q = queue.Queue()
_hot_syms = deque(maxlen=30)
_macro = {"btc": "UNKNOWN"}
_ks = {"active": False, "reason": "", "resume": 0, "consec": 0, "daily": 0.0, "day_reset": 0}
_stats = {
    "trades": 0, "wins": 0, "losses": 0, "pnl": 0.0,
    "best": 0.0, "worst": 0.0,
    "tp_hit": 0, "sl_hit": 0, "trail": 0,
    "hist": deque(maxlen=200), "start": time.time(),
}

# ═══════════════════════════════════════════════════════════════════════════
#  FUNGSI CORE – PERBAIKAN UTAMA DI live_open & monitor_positions
# ═══════════════════════════════════════════════════════════════════════════

live_positions = {}
trade_log = []
signal_weights = SignalWeights()
scorer = SignalScorer(signal_weights)
learning = LearningLayer(signal_weights)

def live_open(sym, direction, score, sigs, price, atr, regime, bias):
    with _lock:
        if sym in live_positions or len(live_positions) >= MAX_POSITIONS:
            return
        live_positions[sym] = {"_r": True}

    px_now = price_live(sym)
    if px_now > 0:
        if abs(px_now - price) / price > SLIPPAGE_GUARD:
            with _lock: live_positions.pop(sym, None)
            return
        price = px_now

    try:
        q_val = qty(sym, price)
    except:
        with _lock: live_positions.pop(sym, None)
        return

    # ── Hitung TP & SL sesuai mode ────────────────────────────────────
    if REVERSED_MODE:
        tp_price, sl_price = RiskManager.calculate_reversed_tpsl(price, direction)
        sl_pct = EXTREME_TP_PCT  # 2.00% (besar)
        tp_pct = HARD_SL_PCT     # 0.50% (kecil)
    else:
        sl_price, tp_price, sl_pct, tp_pct = RiskManager.calculate_sl_tp(price, direction)

    pos = {
        "side":       direction,
        "entry":      price,
        "qty":        q_val,
        "open_time":  time.time(),
        "score":      score,
        "sigs":       sigs,
        "atr":        atr,
        "tp_price":   tp_price,
        "sl_price":   sl_price,
        "tp_pct":     tp_pct,
        "sl_pct":     sl_pct,
        "regime":     regime,
        "bias":       bias,
        "trail_active": False,
        "best_price":   price,
    }
    with _lock:
        live_positions[sym] = pos

    d = "🟢" if direction == "LONG" else "🔴"
    print(f"\n  {d} [DRY] {sym} {direction} @{price:.6g} "
          f"| TP:{tp_price:.6g} SL:{sl_price:.6g} "
          f"| Regime:{regime}")
    if REVERSED_MODE:
        print(f"        [REVERSED] TP=0.50% SL=2.00% | Signals: {' | '.join(sigs[:5])}")
    else:
        print(f"        Trail ≥0.40% | Signals: {' | '.join(sigs[:5])}")
    _stats["trades"] += 1

def live_close(sym, reason, price=None):
    with _lock:
        pos = live_positions.pop(sym, None)
    if pos is None or pos.get("_r"):
        return

    if price is None:
        price = price_live(sym)
    if price == 0:
        return

    side, entry, q_val = pos["side"], pos["entry"], pos["qty"]
    gross_pnl = (price - entry) * q_val if side == "LONG" else (entry - price) * q_val
    notional_entry = entry * q_val
    notional_exit  = price * q_val
    total_fee = (notional_entry + notional_exit) * FEE_RATE
    pnl = gross_pnl - total_fee

    gross_pct = (price - entry) / entry * 100 if side == "LONG" else (entry - price) / entry * 100
    hold = time.time() - pos["open_time"]
    won = pnl >= 0
    e = "🟢" if won else "🔴"

    print(f"  {e} [DRY] {sym} {side} CLOSE — {reason}")
    print(f"     {entry:.6g}→{price:.6g} ({gross_pct:+.3f}% gross) "
          f"hold:{hold:.0f}s | PnL:{pnl:+.5f}U (fee:{total_fee:.5f}U)")

    trade = TradeRecord(symbol=sym, direction=side, entry_price=entry, exit_price=price,
                        pnl=pnl, won=won, regime=pos.get("regime", "UNKNOWN"),
                        signals=pos.get("sigs", []), score=pos.get("score", 0),
                        atr_entry=pos.get("atr", 0),
                        sl_pct=pos.get("sl_pct", 0), tp_pct=pos.get("tp_pct", 0),
                        hold_seconds=hold, exit_reason=reason)
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

    if "TakeProfit" in reason or "TP" in reason:
        _stats["tp_hit"] += 1
    elif "StopLoss" in reason or "SL" in reason:
        _stats["sl_hit"] += 1
    elif "Trail" in reason:
        _stats["trail"] += 1

    trade_log.append({
        "sym": sym, "side": side, "entry": round(entry, 7), "exit": round(price, 7),
        "pnl": round(pnl, 5), "reason": reason, "hold": int(hold),
    })
    _hot_syms.appendleft(sym)
    _rescan_q.put(1)
    print_inline()

def monitor_positions():
    for sym in list(live_positions.keys()):
        pos = live_positions.get(sym)
        if pos is None or pos.get("_r"):
            continue

        px = price_live(sym)
        if px == 0:
            continue

        side = pos["side"]
        entry = pos["entry"]
        tp = pos["tp_price"]
        sl = pos["sl_price"]

        # ── Cek TP (selalu dicek lebih dulu) ────────────────────────
        if side == "LONG" and px >= tp:
            live_close(sym, f"TakeProfit ({pos['tp_pct']*100:.2f}%)", px)
            continue
        if side == "SHORT" and px <= tp:
            live_close(sym, f"TakeProfit ({pos['tp_pct']*100:.2f}%)", px)
            continue

        # ── Cek SL ──────────────────────────────────────────────────
        if side == "LONG" and px <= sl:
            live_close(sym, f"StopLoss ({pos['sl_pct']*100:.2f}%)", px)
            continue
        if side == "SHORT" and px >= sl:
            live_close(sym, f"StopLoss ({pos['sl_pct']*100:.2f}%)", px)
            continue

        # ── Trailing hanya jika mode normal ─────────────────────────
        if not REVERSED_MODE:
            # ... kode trailing sama seperti sebelumnya
            pass

# ═══════════════════════════════════════════════════════════════════════════
#  SCANNER & THREADS (TIDAK BERUBAH KECUALI REVERSE LOGIC)
# ═══════════════════════════════════════════════════════════════════════════

def scan_one(sym):
    # ... (ambil sinyal, lalu jika REVERSED_MODE balik arah)
    # PENTING: balik arah DI SINI, sebelum live_open
    pass

# ═══════════════════════════════════════════════════════════════════════════
#  PRINT & MAIN (SESUAIKAN LABEL TP/SL)
# ═══════════════════════════════════════════════════════════════════════════

def print_inline():
    n = _stats["wins"] + _stats["losses"]
    wr = _stats["wins"] / n * 100 if n else 0
    pnl = _stats["pnl"]
    e = "💚" if pnl >= 0 else "🔴"
    mode = "[REVERSED]" if REVERSED_MODE else "[NORMAL]"
    print(f"       ┌ {mode} {n}T WR:{wr:.0f}% W:{_stats['wins']} L:{_stats['losses']} {e}PnL:{pnl:+.4f}U")
    if REVERSED_MODE:
        print(f"       └ TP(0.50%):{_stats['tp_hit']} SL(2.00%):{_stats['sl_hit']} Trail:OFF")
    else:
        print(f"       └ XTP:{_stats['sl_hit']} Trail:{_stats['trail']} HardSL:{_stats['tp_hit']}")

def print_full():
    # ... (sesuaikan dengan statistik tp_hit / sl_hit)
    pass

def run_bot():
    # ... (tampilkan header REVERSED)
    pass

if __name__ == "__main__":
    run_bot()
