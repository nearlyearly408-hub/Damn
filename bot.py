"""
Bot Scalping v20.1.1 — TRAILING-STOP PROFIT-LOCK + ANTI-RUGI FEE
=================================================================
PERUBAHAN dari v20.1.0:
1. Trailing stop sekarang MURNI untuk mengunci profit:
   - Mulai aktif saat gross profit >= 0.15% -> stop di ENTRY (breakeven)
   - Semakin profit naik, stop ikut naik (dari best price)
   - Stop TIDAK PERNAH lebih rendah dari entry + sedikit margin fee
2. MIN_PROFIT_LOCK 0.05% memastikan exit selalu > fee (net positif kecil)
3. TRAIL_PHASES dirapihkan agar lebih lebar di awal, lebih ketat saat profit besar
4. Reversed logic DIMATIKAN sementara — silakan nyalakan jika ingin
5. MIN_SCORE dinaikkan ke 60 (range/volatile) dan 55 (trending)
6. Kill Switch daily loss dikecilkan ke -8U agar lebih protektif

Target: WR > 40%, Profit Factor > 1.2, Max Drawdown < -5U
"""

import os
import time
import math
import threading
import queue
import json
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

# ── Risk Management (FIXED PCT) ─────────────────────────────────────────
HARD_SL_PCT       = 0.004    # 0.40% price move → max loss ~$0.16
EXTREME_TP_PCT    = 0.015    # 1.50% backstop TP
FEE_RATE          = 0.001    # 0.10% per leg

# ── Progressive Trailing Stop (PROFIT-LOCK ONLY) ──────────────────────
# Format: (gross_profit_threshold, trailing_gap)
# gap = jarak stop dari BEST PRICE (bukan dari entry)
# STOP TIDAK AKAN PERNAH LEBIH RENDAH DARI ENTRY + MIN_PROFIT_LOCK
TRAIL_PHASES = [
    (0.0060, 0.0040),   # profit >= 0.60% → gap 0.40% (net stop ≈ +0.20%)
    (0.0040, 0.0025),   # profit >= 0.40% → gap 0.25% (net stop ≈ +0.15%)
    (0.0025, 0.0015),   # profit >= 0.25% → gap 0.15% (net stop ≈ +0.10%)
    (0.0015, 0.0015),   # profit >= 0.15% → gap 0.15% (net stop ≈ +0.00% = breakeven)
]
# TRAIL_PHASES harus diurutkan dari threshold TERBESAR ke TERKECIL

# ── Minimal profit lock setelah trailing aktif ───────────────────────
MIN_PROFIT_LOCK = 0.0005   # 0.05% di atas entry agar nutup fee (net ≈ +0.03% setelah fee)

# ═══════════════════════════════════════════════════════════════════════════
#  SYMBOLS
# ═══════════════════════════════════════════════════════════════════════════

SYMBOLS = [
    "BTCUSDT",  "ETHUSDT",  "BNBUSDT",  "SOLUSDT",  "XRPUSDT",
    "ADAUSDT",  "DOGEUSDT", "AVAXUSDT", "TRXUSDT",  "DOTUSDT",
    "LINKUSDT", "MATICUSDT","LTCUSDT",  "ATOMUSDT", "UNIUSDT",
    "NEARUSDT", "APTUSDT",  "ARBUSDT",  "OPUSDT",   "INJUSDT",
    "SUIUSDT",  "SEIUSDT",  "FETUSDT",  "WLDUSDT",  "AAVEUSDT",
    "ORDIUSDT", "TONUSDT",  "1000PEPEUSDT","WIFUSDT","JUPUSDT",
    "FTMUSDT",  "SANDUSDT", "MANAUSDT", "GALAUSDT", "APEUSDT",
    "CRVUSDT",  "1000SHIBUSDT","COMPUSDT","MKRUSDT", "SNXUSDT",
]
SYMBOLS = list(dict.fromkeys(SYMBOLS))

# ── Scanning ──────────────────────────────────────────────────────────────
SCAN_INTERVAL = 2.0
MONITOR_INT   = 0.5
BATCH_SIZE    = 15
MAX_WORKERS   = 5
SLOT_FILL_INT = 0.01

# ── Scoring & Filter ──────────────────────────────────────────────────────
MIN_SCORE      = 55       # untuk trending regime
MIN_SCORE_RANGE = 60      # untuk range/volatile/exhaustion
MIN_GAP        = 10
SLIPPAGE_GUARD = 0.0015
TTL_5M         = 2

# ── Kill Switch ───────────────────────────────────────────────────────────
DAILY_LOSS  = -8.0        # lebih ketat agar tidak jebol
CONSEC_MAX  = 12
CONSEC_PAUSE = 15

# ── Learning ──────────────────────────────────────────────────────────────
LEARNING_WINDOW      = 200
MIN_TRADES_FOR_WEIGHT = 20

# ═══════════════════════════════════════════════════════════════════════════
#  MARKET REGIME DETECTION
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
        atr, atr_prev    = row["atr"], prev["atr"]
        adx              = row["adx"]

        bull_stack  = close > e5 > e9 > e21 > e50
        bear_stack  = close < e5 < e9 < e21 < e50
        mild_bull   = close > e9 > e21
        mild_bear   = close < e9 < e21

        strong_trend      = adx > 25
        very_strong_trend = adx > 35

        atr_expand  = (atr / atr_prev) > 1.2 if atr_prev > 0 else False
        atr_collapse = (atr / atr_prev) < 0.8 if atr_prev > 0 else False

        m5         = row["m5"]
        m5_prev    = prev["m5"]
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
        elif (atr_collapse and decelerating) or (20 < adx < 35 and decelerating):
            bias = 1 if m5 > 0 else -1
            return MarketRegime.REGIME_EXHAUSTION, 40, bias
        else:
            return MarketRegime.REGIME_RANGE, 30, 0

# ═══════════════════════════════════════════════════════════════════════════
#  EXHAUSTION CONFIRMATION LAYER
# ═══════════════════════════════════════════════════════════════════════════

class ExhaustionConfirmation:
    @staticmethod
    def check_short_exhaustion(df: pd.DataFrame) -> Tuple[bool, int, List[str]]:
        if df is None or len(df) < 55:
            return False, 0, []
        row, prev = df.iloc[-2], df.iloc[-3]
        conditions, reasons = [], []

        if row["rsi"] > 75:
            conditions.append(True); reasons.append(f"RSI_{row['rsi']:.0f}>75")
        else: conditions.append(False)

        high_price = max(df["high"].iloc[-10:])
        high_rsi   = max(df["rsi"].iloc[-10:])
        if row["close"] >= high_price * 0.99 and row["rsi"] < high_rsi - 3:
            conditions.append(True); reasons.append("RSI_Div")
        else: conditions.append(False)

        high_macd = max(df["mh"].iloc[-10:])
        if row["close"] >= high_price * 0.99 and row["mh"] < high_macd - 0.5 * row["atr"]:
            conditions.append(True); reasons.append("MACD_Div")
        else: conditions.append(False)

        if row["vr"] > 2.0:
            conditions.append(True); reasons.append(f"VolClimax_{row['vr']:.1f}x")
        else: conditions.append(False)

        vol_prev = prev["vr"] if not np.isnan(prev["vr"]) else 1
        if row["vr"] > 1.8 and row["vr"] > vol_prev * 1.2:
            conditions.append(True); reasons.append("DeltaVolClimax")
        else: conditions.append(False)

        body       = abs(row["close"] - row["open"])
        upper_wick = row["high"] - max(row["close"], row["open"])
        if upper_wick > body * 1.5 and upper_wick > row["atr"] * 0.3:
            conditions.append(True); reasons.append("LongUpperWick")
        else: conditions.append(False)

        atr_series = df["atr"].iloc[-10:]
        atr_peak   = atr_series.max()
        if atr_peak > atr_series.iloc[-5] * 1.3 and row["atr"] < atr_peak * 0.8:
            conditions.append(True); reasons.append("ATR_ExpCollapse")
        else: conditions.append(False)

        m5, m5_prev = row["m5"], prev["m5"]
        if m5 > 0.002 and m5 < m5_prev * 0.7:
            conditions.append(True); reasons.append("MomDecel")
        else: conditions.append(False)

        br_peak = max(df["br"].iloc[-10:])
        if row["br"] < br_peak - 0.1 and br_peak > 0.6:
            conditions.append(True); reasons.append("OrderflowRev")
        else: conditions.append(False)

        count = sum(conditions)
        return count >= 3, count, reasons

    @staticmethod
    def check_long_exhaustion(df: pd.DataFrame) -> Tuple[bool, int, List[str]]:
        if df is None or len(df) < 55:
            return False, 0, []
        row, prev = df.iloc[-2], df.iloc[-3]
        conditions, reasons = [], []

        if row["rsi"] < 25:
            conditions.append(True); reasons.append(f"RSI_{row['rsi']:.0f}<25")
        else: conditions.append(False)

        low_price = min(df["low"].iloc[-10:])
        low_rsi   = min(df["rsi"].iloc[-10:])
        if row["close"] <= low_price * 1.01 and row["rsi"] > low_rsi + 3:
            conditions.append(True); reasons.append("RSI_Div_Bull")
        else: conditions.append(False)

        low_macd = min(df["mh"].iloc[-10:])
        if row["close"] <= low_price * 1.01 and row["mh"] > low_macd + 0.5 * row["atr"]:
            conditions.append(True); reasons.append("MACD_Div_Bull")
        else: conditions.append(False)

        if row["vr"] > 2.0:
            conditions.append(True); reasons.append(f"VolClimax_{row['vr']:.1f}x")
        else: conditions.append(False)

        vol_prev = prev["vr"] if not np.isnan(prev["vr"]) else 1
        if row["vr"] > 1.8 and row["vr"] > vol_prev * 1.2:
            conditions.append(True); reasons.append("DeltaVolClimax")
        else: conditions.append(False)

        body       = abs(row["close"] - row["open"])
        lower_wick = min(row["close"], row["open"]) - row["low"]
        if lower_wick > body * 1.5 and lower_wick > row["atr"] * 0.3:
            conditions.append(True); reasons.append("LongLowerWick")
        else: conditions.append(False)

        atr_series = df["atr"].iloc[-10:]
        atr_peak   = atr_series.max()
        if atr_peak > atr_series.iloc[-5] * 1.3 and row["atr"] < atr_peak * 0.8:
            conditions.append(True); reasons.append("ATR_ExpCollapse")
        else: conditions.append(False)

        m5, m5_prev = row["m5"], prev["m5"]
        if m5 < -0.002 and m5 > m5_prev * 0.7:
            conditions.append(True); reasons.append("MomDecel_Bull")
        else: conditions.append(False)

        br_trough = min(df["br"].iloc[-10:])
        if row["br"] > br_trough + 0.1 and br_trough < 0.4:
            conditions.append(True); reasons.append("OrderflowRev_Bull")
        else: conditions.append(False)

        count = sum(conditions)
        return count >= 3, count, reasons

# ═══════════════════════════════════════════════════════════════════════════
#  SELF-LEARNING SIGNAL WEIGHTING
# ═══════════════════════════════════════════════════════════════════════════

class SignalWeights:
    def __init__(self):
        self.weights = {
            "ema_bull_stack": 35,  "ema_mild_bull": 26,       "ema_weak_bull": 14,
            "mom_strong": 30,      "mom_moderate": 20,
            "macd_cross_up": 22,   "macd_strengthen": 15,
            "orderflow_buy_climax": 25, "orderflow_buy_high": 14,
            "rsi_extreme_ob": 25,  "rsi_high": 12,
            "ema_bear_stack": 35,  "ema_mild_bear": 26,       "ema_weak_bear": 14,
            "mom_strong_neg": 30,  "mom_moderate_neg": 20,
            "macd_cross_down": 22, "macd_strengthen_neg": 15,
            "orderflow_sell_climax": 25, "orderflow_sell_high": 14,
            "rsi_extreme_os": 25,  "rsi_low": 12,
        }
        self.history         = defaultdict(list)
        self.adaptive_enabled = True

    def record_outcome(self, signals: List[str], won: bool):
        for sig in signals:
            base = sig.split('[')[0].strip()
            if base in self.weights:
                self.history[base].append(1 if won else 0)
                if len(self.history[base]) > LEARNING_WINDOW:
                    self.history[base] = self.history[base][-LEARNING_WINDOW:]

    def get_adjusted_weight(self, signal_name: str) -> float:
        if not self.adaptive_enabled:
            return self.weights.get(signal_name, 10)
        base = signal_name.split('[')[0].strip()
        hist = self.history.get(base, [])
        if len(hist) < MIN_TRADES_FOR_WEIGHT:
            return self.weights.get(base, 10)
        win_rate = sum(hist) / len(hist)
        factor   = max(0.5, min(1.5, 0.5 + win_rate))
        return self.weights.get(base, 10) * factor

# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL SCORING WITH REGIME & EXHAUSTION
# ═══════════════════════════════════════════════════════════════════════════

class SignalScorer:
    def __init__(self, signal_weights: SignalWeights):
        self.weights = signal_weights

    def get_signal(self, df: pd.DataFrame, symbol: str = None):
        if df is None or len(df) < 55:
            return None, 0, [], 0.0, 0.0, 0.0, "UNKNOWN", 0.0

        regime, strength, bias = MarketRegime.detect(df)
        long_score,  long_sigs  = self._score_long(df)
        short_score, short_sigs = self._score_short(df)

        atr = df["atr"].iloc[-2]

        is_exh_short, cnt_short, rsn_short = False, 0, []
        is_exh_long,  cnt_long,  rsn_long  = False, 0, []

        if regime in (MarketRegime.REGIME_RANGE,
                      MarketRegime.REGIME_EXHAUSTION,
                      MarketRegime.REGIME_VOLATILE):
            is_exh_short, cnt_short, rsn_short = \
                ExhaustionConfirmation.check_short_exhaustion(df)
            is_exh_long, cnt_long, rsn_long = \
                ExhaustionConfirmation.check_long_exhaustion(df)

        # ── Penentuan arah dengan score minimum berbeda per regime ──
        min_score = MIN_SCORE_RANGE if regime in (
            MarketRegime.REGIME_RANGE,
            MarketRegime.REGIME_VOLATILE,
            MarketRegime.REGIME_EXHAUSTION
        ) else MIN_SCORE

        if regime == MarketRegime.REGIME_TRENDING_BULL:
            if long_score >= min_score:
                return "LONG", long_score, long_sigs, atr, 0, 0, regime, bias
            return None, max(long_score, short_score), [], atr, 0, 0, regime, bias

        elif regime == MarketRegime.REGIME_TRENDING_BEAR:
            if short_score >= min_score:
                return "SHORT", short_score, short_sigs, atr, 0, 0, regime, bias
            return None, max(long_score, short_score), [], atr, 0, 0, regime, bias

        elif regime == MarketRegime.REGIME_RANGE:
            if short_score > long_score and short_score >= min_score and is_exh_short:
                return "SHORT", short_score, short_sigs + rsn_short, atr, 0, 0, regime, bias
            if long_score > short_score and long_score >= min_score and is_exh_long:
                return "LONG", long_score, long_sigs + rsn_long, atr, 0, 0, regime, bias
            return None, max(long_score, short_score), [], atr, 0, 0, regime, bias

        elif regime == MarketRegime.REGIME_EXHAUSTION:
            if short_score > long_score and short_score >= min_score and cnt_short >= 2:
                return "SHORT", short_score, short_sigs + rsn_short, atr, 0, 0, regime, bias
            if long_score > short_score and long_score >= min_score and cnt_long >= 2:
                return "LONG", long_score, long_sigs + rsn_long, atr, 0, 0, regime, bias
            return None, max(long_score, short_score), [], atr, 0, 0, regime, bias

        elif regime == MarketRegime.REGIME_VOLATILE:
            if short_score > long_score and short_score >= min_score + 5 and is_exh_short:
                return "SHORT", short_score, short_sigs + rsn_short, atr, 0, 0, regime, bias
            if long_score > short_score and long_score >= min_score + 5 and is_exh_long:
                return "LONG", long_score, long_sigs + rsn_long, atr, 0, 0, regime, bias
            return None, max(long_score, short_score), [], atr, 0, 0, regime, bias

        return None, 0, [], atr, 0, 0, regime, bias

    def _score_long(self, df: pd.DataFrame) -> Tuple[int, List[str]]:
        row, prev, prev2 = df.iloc[-2], df.iloc[-3], df.iloc[-4]
        score, signals = 0, []
        p, e5, e9, e21, e50 = row["close"], row["e5"], row["e9"], row["e21"], row["e50"]

        if   p < e5 < e9 < e21 < e50:
            w = self.weights.get_adjusted_weight("ema_bear_stack");  score += w; signals.append(f"EMA5↓[{w:.0f}]")
        elif p < e5 < e9 < e21:
            w = self.weights.get_adjusted_weight("ema_mild_bear");   score += w; signals.append(f"EMA4↓[{w:.0f}]")
        elif p < e5 < e9:
            w = self.weights.get_adjusted_weight("ema_weak_bear");   score += w; signals.append(f"EMA3↓[{w:.0f}]")

        m5 = row["m5"]
        if   m5 < -0.003:
            w = self.weights.get_adjusted_weight("mom_strong_neg");   score += w; signals.append(f"Mom{m5*100:.1f}%↓[{w:.0f}]")
        elif m5 < -0.002:
            w = self.weights.get_adjusted_weight("mom_moderate_neg"); score += w; signals.append(f"Mom{m5*100:.1f}%↓[{w:.0f}]")

        mh, mh_p, mh_p2 = row["mh"], prev["mh"], prev2["mh"]
        if   mh_p >= 0 and mh < 0:
            w = self.weights.get_adjusted_weight("macd_cross_down");    score += w; signals.append(f"MACD_X↓[{w:.0f}]")
        elif mh < 0 and mh < mh_p < mh_p2:
            w = self.weights.get_adjusted_weight("macd_strengthen_neg"); score += w; signals.append(f"MACD↓↓[{w:.0f}]")

        br = row["br"]
        if   br < 0.44:
            w = self.weights.get_adjusted_weight("orderflow_sell_climax"); score += w; signals.append(f"SellClimax{1-br:.0%}[{w:.0f}]")
        elif br < 0.48:
            w = self.weights.get_adjusted_weight("orderflow_sell_high");   score += w; signals.append(f"Sell{1-br:.0%}[{w:.0f}]")

        rsi = row["rsi"]
        if   rsi < 32:
            w = self.weights.get_adjusted_weight("rsi_extreme_os"); score += w; signals.append(f"RSI{rsi:.0f}OS[{w:.0f}]")
        elif rsi < 40:
            w = self.weights.get_adjusted_weight("rsi_low");        score += w; signals.append(f"RSI{rsi:.0f}Lo[{w:.0f}]")

        return score, signals

    def _score_short(self, df: pd.DataFrame) -> Tuple[int, List[str]]:
        row, prev, prev2 = df.iloc[-2], df.iloc[-3], df.iloc[-4]
        score, signals = 0, []
        p, e5, e9, e21, e50 = row["close"], row["e5"], row["e9"], row["e21"], row["e50"]

        if   p > e5 > e9 > e21 > e50:
            w = self.weights.get_adjusted_weight("ema_bull_stack");  score += w; signals.append(f"EMA5↑[{w:.0f}]")
        elif p > e5 > e9 > e21:
            w = self.weights.get_adjusted_weight("ema_mild_bull");   score += w; signals.append(f"EMA4↑[{w:.0f}]")
        elif p > e5 > e9:
            w = self.weights.get_adjusted_weight("ema_weak_bull");   score += w; signals.append(f"EMA3↑[{w:.0f}]")

        m5 = row["m5"]
        if   m5 > 0.003:
            w = self.weights.get_adjusted_weight("mom_strong");   score += w; signals.append(f"Mom+{m5*100:.1f}%↑[{w:.0f}]")
        elif m5 > 0.002:
            w = self.weights.get_adjusted_weight("mom_moderate"); score += w; signals.append(f"Mom+{m5*100:.1f}%↑[{w:.0f}]")

        mh, mh_p, mh_p2 = row["mh"], prev["mh"], prev2["mh"]
        if   mh_p <= 0 and mh > 0:
            w = self.weights.get_adjusted_weight("macd_cross_up");    score += w; signals.append(f"MACD_X↑[{w:.0f}]")
        elif mh > 0 and mh > mh_p > mh_p2:
            w = self.weights.get_adjusted_weight("macd_strengthen"); score += w; signals.append(f"MACD↑↑[{w:.0f}]")

        br = row["br"]
        if   br > 0.56:
            w = self.weights.get_adjusted_weight("orderflow_buy_climax"); score += w; signals.append(f"BuyClimax{br:.0%}[{w:.0f}]")
        elif br > 0.52:
            w = self.weights.get_adjusted_weight("orderflow_buy_high");   score += w; signals.append(f"Buy{br:.0%}[{w:.0f}]")

        rsi = row["rsi"]
        if   rsi > 68:
            w = self.weights.get_adjusted_weight("rsi_extreme_ob"); score += w; signals.append(f"RSI{rsi:.0f}OB[{w:.0f}]")
        elif rsi > 60:
            w = self.weights.get_adjusted_weight("rsi_high");       score += w; signals.append(f"RSI{rsi:.0f}Hi[{w:.0f}]")

        return score, signals

# ═══════════════════════════════════════════════════════════════════════════
#  RISK MANAGER v2 — FIXED PCT + TRAILING PROFIT LOCK
# ═══════════════════════════════════════════════════════════════════════════

class RiskManager:
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
    def get_trail_stop_price(entry_price: float, direction: str,
                             best_price: float, best_gross_pct: float) -> Optional[Tuple[float, float]]:
        """
        Cek apakah trailing stop aktif.
        Returns: (stop_price, gap) jika trailing aktif, None jika belum.
        STOP PRICE TIDAK PERNAH LEBIH BURUK DARI ENTRY + MIN_PROFIT_LOCK
        """
        for threshold, gap in TRAIL_PHASES:
            if best_gross_pct >= threshold:
                # Hitung stop price dari best price
                if direction == "LONG":
                    stop = best_price * (1 - gap)
                    # Pastikan tidak di bawah entry + lock profit
                    min_stop = entry_price * (1 + MIN_PROFIT_LOCK)
                    if stop < min_stop:
                        stop = min_stop
                else:  # SHORT
                    stop = best_price * (1 + gap)
                    min_stop = entry_price * (1 - MIN_PROFIT_LOCK)
                    if stop > min_stop:
                        stop = min_stop
                return stop, gap
        return None  # trailing belum aktif, pakai HardSL

# ═══════════════════════════════════════════════════════════════════════════
#  TRADE RECORDER & LEARNING LAYER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    symbol:       str
    direction:    str
    entry_price:  float
    exit_price:   float
    pnl:          float
    won:          bool
    regime:       str
    signals:      List[str]
    score:        float
    atr_entry:    float
    sl_pct:       float
    tp_pct:       float
    hold_seconds: float
    exit_reason:  str = ""
    timestamp:    float = field(default_factory=time.time)

class LearningLayer:
    def __init__(self, signal_weights: SignalWeights):
        self.sw                  = signal_weights
        self.trades: List[TradeRecord] = []
        self.stats_by_regime     = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})
        self.stats_by_sym        = defaultdict(lambda: {"wins": 0, "losses": 0})

    def add_trade(self, t: TradeRecord):
        self.trades.append(t)
        s = self.stats_by_regime[t.regime]
        s["wins"]   += 1 if t.won else 0
        s["losses"] += 0 if t.won else 1
        s["pnl"]    += t.pnl
        self.stats_by_sym[t.symbol]["wins"]   += 1 if t.won else 0
        self.stats_by_sym[t.symbol]["losses"] += 0 if t.won else 1
        self.sw.record_outcome(t.signals, t.won)
        if len(self.trades) > 1000:
            self.trades = self.trades[-500:]

    def winrate(self, regime: str = None) -> float:
        if regime:
            s = self.stats_by_regime[regime]
            t = s["wins"] + s["losses"]
            return s["wins"] / t if t else 0.5
        wins   = sum(s["wins"]   for s in self.stats_by_regime.values())
        losses = sum(s["losses"] for s in self.stats_by_regime.values())
        total  = wins + losses
        return wins / total if total else 0.5

# ═══════════════════════════════════════════════════════════════════════════
#  BOT STATE & UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

_precision_cache: Dict[str, int] = {}
_ohlcv_cache:     Dict           = {}
_ticker_cache:    Dict           = {}
_ticker_ts        = 0.0
_lock             = threading.Lock()
_executor         = ThreadPoolExecutor(max_workers=MAX_WORKERS)
_rescan_q         = queue.Queue()
_hot_syms         = deque(maxlen=30)

_macro = {"btc": "UNKNOWN"}
_ks    = {"active": False, "reason": "", "resume": 0, "consec": 0,
          "daily": 0.0, "day_reset": 0}
_stats = {
    "trades": 0, "wins": 0, "losses": 0, "pnl": 0.0, "best": 0.0, "worst": 0.0,
    "extreme_tp": 0, "hard_sl": 0, "trail_stop": 0,
    "hist": deque(maxlen=200), "start": time.time(),
}

def get_precision(symbol: str) -> int:
    if symbol in _precision_cache:
        return _precision_cache[symbol]
    try:
        info = client.futures_exchange_info()
        for s in info['symbols']:
            if s['symbol'] == symbol:
                prec = int(s['quantityPrecision'])
                _precision_cache[symbol] = prec
                return prec
    except:
        pass
    return 2

def qty(symbol: str, price: float) -> float:
    raw  = (ORDER_USDT * LEVERAGE) / price
    prec = get_precision(symbol)
    return round(raw, prec)

def price_live(symbol: str) -> float:
    try:
        return float(client.futures_symbol_ticker(symbol=symbol)["price"])
    except:
        return 0.0

def tickers_all() -> Dict:
    global _ticker_cache, _ticker_ts
    now = time.time()
    if now - _ticker_ts < 2 and _ticker_cache:
        return _ticker_cache
    try:
        raw = client.futures_ticker()
        _ticker_cache = {
            t["symbol"]: {
                "pct":  float(t["priceChangePercent"]),
                "vol":  float(t["quoteVolume"]),
                "last": float(t["lastPrice"])
            } for t in raw
        }
        _ticker_ts = now
    except:
        pass
    return _ticker_cache

def ohlcv(symbol: str, interval: str, limit: int = 100) -> Optional[pd.DataFrame]:
    key, now = (symbol, interval), time.time()
    if key in _ohlcv_cache and now - _ohlcv_cache[key][0] < TTL_5M:
        return _ohlcv_cache[key][1]
    try:
        kl = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(kl, columns=["time","open","high","low","close","volume",
                                        "ct","qv","trades","tbbase","tbquote","ignore"])
        for c in ["open","high","low","close","volume","tbbase","tbquote"]:
            df[c] = df[c].astype(float)
        df = _add_ta(df)
        _ohlcv_cache[key] = (now, df)
        return df
    except:
        cached = _ohlcv_cache.get(key)
        return cached[1] if cached else None

def _add_ta(df: pd.DataFrame) -> pd.DataFrame:
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
    df["mh"]  = ta.trend.MACD(df["close"], 12, 26, 9).macd_diff()
    df["e5"]  = ta.trend.EMAIndicator(df["close"], 5).ema_indicator()
    df["e9"]  = ta.trend.EMAIndicator(df["close"], 9).ema_indicator()
    df["e21"] = ta.trend.EMAIndicator(df["close"], 21).ema_indicator()
    df["e50"] = ta.trend.EMAIndicator(df["close"], 50).ema_indicator()
    df["atr"] = ta.volatility.AverageTrueRange(
                    df["high"], df["low"], df["close"], 14).average_true_range()
    df["adx"] = ta.trend.ADXIndicator(
                    df["high"], df["low"], df["close"], 14).adx()
    df["vm"]  = df["volume"].rolling(20).mean()
    df["vr"]  = df["volume"] / df["vm"].replace(0, 1)
    df["br"]  = df["tbbase"] / df["volume"].replace(0, 1)
    df["body"] = abs(df["close"] - df["open"])
    df["rng"]  = df["high"] - df["low"]
    df["br2"]  = df["body"] / df["rng"].replace(0, 1)
    df["m5"]   = (df["close"] - df["close"].shift(5)) / df["close"].shift(5)
    df["m3"]   = (df["close"] - df["close"].shift(3)) / df["close"].shift(3)
    return df

def ks_check() -> Tuple[bool, str]:
    k, now = _ks, time.time()
    if k["active"] and now >= k["resume"]:
        k["active"] = False; k["consec"] = 0
    if k["active"]:
        return True, k["reason"]
    day = now - (now % 86400)
    if day > k["day_reset"]:
        k["daily"] = 0.0; k["day_reset"] = day
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

def ks_upd(pnl: float):
    _ks["daily"] += pnl
    _ks["consec"] = 0 if pnl >= 0 else _ks["consec"] + 1

# ═══════════════════════════════════════════════════════════════════════════
#  TRADING STATE
# ═══════════════════════════════════════════════════════════════════════════

live_positions: Dict = {}
trade_log:      List = []
signal_weights = SignalWeights()
scorer         = SignalScorer(signal_weights)
learning       = LearningLayer(signal_weights)

# ═══════════════════════════════════════════════════════════════════════════
#  CORE TRADING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def live_open(sym, direction, score, sigs, price, atr, regime, bias):
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

    sl_price, tp_price, sl_pct, tp_pct = RiskManager.calculate_sl_tp(price, direction)

    pos = {
        "side":       direction,
        "entry":      price,
        "qty":        q_val,
        "open_time":  time.time(),
        "score":      score,
        "sigs":       sigs,
        "atr":        atr,
        "sl_price":   sl_price,
        "tp_price":   tp_price,
        "sl_pct":     sl_pct,
        "tp_pct":     tp_pct,
        "regime":     regime,
        "bias":       bias,
        "trail_active": False,
        "trail_stop":   None,
        "trail_phase":  None,
        "best_price":   price,
    }
    with _lock:
        live_positions[sym] = pos

    d = "🟢" if direction == "LONG" else "🔴"
    print(f"\n  {d} [DRY] {sym} {direction} @{price:.6g} "
          f"| HardSL:{sl_pct*100:.2f}% XTP:{tp_pct*100:.2f}% "
          f"| Regime:{regime}")
    print(f"        Trail aktif ≥0.15% gross (breakeven lock) | Signals: {' | '.join(sigs[:5])}")
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
    gross_pnl = ((price - entry) * q_val if side == "LONG"
                 else (entry - price) * q_val)

    notional_entry = entry * q_val
    notional_exit  = price * q_val
    total_fee      = (notional_entry + notional_exit) * FEE_RATE
    pnl            = gross_pnl - total_fee

    gross_pct = (price - entry) / entry * 100 if side == "LONG" \
                else (entry - price) / entry * 100
    hold = time.time() - pos["open_time"]
    won  = pnl >= 0
    e    = "🟢" if won else "🔴"

    trail_info = ""
    if pos.get("trail_active") and "Trail" in reason:
        trail_info = f" [gap:{pos.get('trail_phase', 0)*100:.2f}%]"

    print(f"  {e} [DRY] {sym} {side} CLOSE — {reason}{trail_info}")
    print(f"     {entry:.6g}→{price:.6g} ({gross_pct:+.3f}% gross) "
          f"hold:{hold:.0f}s | PnL:{pnl:+.5f}U (fee:{total_fee:.5f}U)")

    trade = TradeRecord(
        symbol=sym, direction=side, entry_price=entry, exit_price=price,
        pnl=pnl, won=won, regime=pos.get("regime", "UNKNOWN"),
        signals=pos.get("sigs", []), score=pos.get("score", 0),
        atr_entry=pos.get("atr", 0),
        sl_pct=pos.get("sl_pct", 0), tp_pct=pos.get("tp_pct", 0),
        hold_seconds=hold, exit_reason=reason,
    )
    learning.add_trade(trade)

    _stats["pnl"]  += pnl
    _stats["hist"].append(pnl)
    ks_upd(pnl)

    if won:
        _stats["wins"] += 1
        if pnl > _stats["best"]: _stats["best"] = pnl
    else:
        _stats["losses"] += 1
        if pnl < _stats["worst"]: _stats["worst"] = pnl

    if "ExtremeTP"  in reason: _stats["extreme_tp"]  += 1
    elif "HardSL"   in reason: _stats["hard_sl"]      += 1
    elif "Trail"    in reason: _stats["trail_stop"]   += 1

    trade_log.append({
        "sym":    sym,
        "side":   side,
        "entry":  round(entry, 7),
        "exit":   round(price, 7),
        "pnl":    round(pnl, 5),
        "reason": reason,
        "hold":   int(hold),
    })
    _hot_syms.appendleft(sym)
    _rescan_q.put(1)
    print_inline()


def monitor_positions():
    for sym in list(live_positions.keys()):
        pos = live_positions.get(sym)
        if pos is None or pos.get("_r"):
            continue

        px   = price_live(sym)
        if px == 0:
            continue

        side    = pos["side"]
        entry   = pos["entry"]
        hard_sl = pos["sl_price"]
        ext_tp  = pos["tp_price"]

        if side == "LONG":
            gross_pct = (px - entry) / entry
        else:
            gross_pct = (entry - px) / entry

        # Update best price
        if side == "LONG":
            if px > pos["best_price"]:
                pos["best_price"] = px
            best = pos["best_price"]
            best_gross = (best - entry) / entry
        else:
            if px < pos["best_price"]:
                pos["best_price"] = px
            best = pos["best_price"]
            best_gross = (entry - best) / entry

        # ── Cek trailing stop (profit lock) ─────────────────────────
        trail_result = RiskManager.get_trail_stop_price(
            entry, side, best, best_gross
        )

        if trail_result is not None:
            trail_stop, gap = trail_result
            pos["trail_active"] = True
            pos["trail_stop"]   = trail_stop
            pos["trail_phase"]  = gap

            if side == "LONG" and px <= trail_stop:
                live_close(sym, f"TrailStop@{gross_pct*100:+.3f}%", px)
                continue
            if side == "SHORT" and px >= trail_stop:
                live_close(sym, f"TrailStop@{gross_pct*100:+.3f}%", px)
                continue
        else:
            pos["trail_active"] = False

        # ── HardSL ──────────────────────────────────────────────────
        if side == "LONG" and px <= hard_sl:
            live_close(sym, "HardSL", px)
            continue
        if side == "SHORT" and px >= hard_sl:
            live_close(sym, "HardSL", px)
            continue

        # ── ExtremeTP ───────────────────────────────────────────────
        if side == "LONG" and px >= ext_tp:
            live_close(sym, "ExtremeTP", px)
            continue
        if side == "SHORT" and px <= ext_tp:
            live_close(sym, "ExtremeTP", px)
            continue

# ═══════════════════════════════════════════════════════════════════════════
#  SCANNER THREAD
# ═══════════════════════════════════════════════════════════════════════════

def scan_one(sym: str):
    try:
        time.sleep(0.002)
        df = ohlcv(sym, Client.KLINE_INTERVAL_5MINUTE, 100)
        if df is None:
            return None

        required = ["rsi","mh","e5","e9","e21","e50","atr","adx","vr","br","m5","br2"]
        if not all(c in df.columns for c in required):
            df = _add_ta(df)

        px  = df["close"].iloc[-2]
        atr = df["atr"].iloc[-2]
        if px == 0 or np.isnan(atr):
            return None

        direction, score, sigs, atr_val, _, _, regime, bias = \
            scorer.get_signal(df, sym)
        if direction is None:
            return None

        # ── REVERSE LOGIC (DINONAKTIFKAN SEMENTARA) ─────────────────
        # Uncomment 2 baris di bawah jika ingin reversed logic kembali
        if direction == "LONG":      direction = "SHORT"; sigs = ["REV_SHORT"] + sigs
        elif direction == "SHORT":   direction = "LONG";  sigs = ["REV_LONG"] + sigs

        px_live = price_live(sym)
        if px_live == 0:
            return None

        return (sym, direction, score, sigs, px_live, atr_val, regime, bias)
    except:
        return None


def scan_batch(syms: List[str]) -> List:
    res = []
    fut = {_executor.submit(scan_one, s): s for s in syms[:BATCH_SIZE]}
    for f in as_completed(fut, timeout=5):
        try:
            r = f.result(timeout=1)
            if r:
                res.append(r)
        except:
            pass
    return res


def top_movers(syms: List[str], n: int = 30) -> List[str]:
    tk, ss = tickers_all(), set(syms)
    mv = [(s, abs(d["pct"])) for s, d in tk.items() if s in ss]
    return [s for s, _ in sorted(mv, key=lambda x: x[1], reverse=True)[:n]]

# ═══════════════════════════════════════════════════════════════════════════
#  PRINTING
# ═══════════════════════════════════════════════════════════════════════════

def print_inline():
    n  = _stats["wins"] + _stats["losses"]
    wr = _stats["wins"] / n * 100 if n else 0
    pnl, e = _stats["pnl"], "💚" if _stats["pnl"] >= 0 else "🔴"
    print(f"       ┌ [v20.1.1 PROFIT-LOCK] {n}T WR:{wr:.0f}% "
          f"W:{_stats['wins']} L:{_stats['losses']} {e}PnL:{pnl:+.4f}U")
    print(f"       └ XTP:{_stats['extreme_tp']} Trail:{_stats['trail_stop']} "
          f"HardSL:{_stats['hard_sl']} | MaxLoss/trade≤$0.20")


def print_full():
    n    = _stats["wins"] + _stats["losses"]
    wr   = _stats["wins"] / n * 100 if n else 0
    pnl  = _stats["pnl"]
    sess = (time.time() - _stats["start"]) / 3600
    tph  = n / sess if sess > 0 else 0
    e    = "💚" if pnl >= 0 else "🔴"

    total_exits = (_stats["extreme_tp"] + _stats["trail_stop"] + _stats["hard_sl"])
    xtp_pct   = _stats["extreme_tp"]  / total_exits * 100 if total_exits else 0
    trail_pct = _stats["trail_stop"]  / total_exits * 100 if total_exits else 0
    sl_pct    = _stats["hard_sl"]     / total_exits * 100 if total_exits else 0

    print(f"\n  {'─'*72}")
    print(f"    ✅ DRY RUN v20.1.1 — TRAILING PROFIT-LOCK + ANTI-FEE LOSS")
    print(f"    🎯 {n}T WR:{wr:.0f}% W:{_stats['wins']} L:{_stats['losses']} ({tph:.1f}T/hr)")
    print(f"    {e} PnL Net:{pnl:+.5f}U Best:{_stats['best']:+.5f} Worst:{_stats['worst']:+.5f}")
    print(f"    💰 Exit breakdown — "
          f"XTP:{_stats['extreme_tp']}({xtp_pct:.0f}%) "
          f"Trail:{_stats['trail_stop']}({trail_pct:.0f}%) "
          f"HardSL:{_stats['hard_sl']}({sl_pct:.0f}%)")
    print(f"    🛡️  Risk: HardSL=0.40% | Trail≥0.15% (breakeven lock) | "
          f"MinLock=0.05% | Fee=0.10%/leg")
    print(f"    📈 Trail phases: ≥0.15%→BE | ≥0.25%→lock0.10% | "
          f"≥0.40%→lock0.15% | ≥0.60%→lock0.20%")
    print(f"    📊 Learning: Global WR {learning.winrate():.1%} | "
          f"Bull WR {learning.winrate('TRENDING_BULL'):.1%}")

    if live_positions:
        print(f"    📌 Open ({len(live_positions)}/{MAX_POSITIONS}):")
        for sym, pos in live_positions.items():
            if pos.get("_r"):
                continue
            px = price_live(sym)
            if px == 0:
                continue
            side  = pos["side"]
            entry = pos["entry"]
            gross = (px - entry) / entry if side == "LONG" else (entry - px) / entry
            trail_info = ""
            if pos.get("trail_active"):
                ts = pos.get("trail_stop", 0)
                gap = pos.get("trail_phase", 0)
                trail_info = f" [TRAIL gap:{gap*100:.2f}% stop:{ts:.6g}]"
            else:
                trail_info = f" [HardSL:{pos['sl_price']:.6g}]"
            print(f"       {sym} {side} entry:{entry:.6g} now:{px:.6g} "
                  f"gross:{gross*100:+.3f}%{trail_info}")

    if trade_log:
        print(f"    📋 Last 5:")
        for t in trade_log[-5:]:
            em = "🟢" if t["pnl"] > 0 else "🔴"
            print(f"       {em} {t['sym']:<16} {t['side']} "
                  f"{t['pnl']:+.5f}U {t['hold']}s — {t['reason']}")
    print(f"  {'─'*72}")

# ═══════════════════════════════════════════════════════════════════════════
#  THREADS
# ═══════════════════════════════════════════════════════════════════════════

def t_monitor():
    while True:
        try:
            if live_positions:
                monitor_positions()
        except:
            pass
        time.sleep(MONITOR_INT)


def t_slot_filler(syms: List[str]):
    scan_idx = 0
    n_bat    = max(1, math.ceil(len(syms) / BATCH_SIZE))
    while True:
        try:
            slots = MAX_POSITIONS - len(live_positions)
            if slots <= 0 or ks_check()[0]:
                time.sleep(SLOT_FILL_INT)
                continue
            hot = [s for s in _hot_syms if s not in live_positions]
            mv  = top_movers(syms, 30)
            mv  = [s for s in mv if s not in live_positions]
            bs  = scan_idx * BATCH_SIZE
            reg = [s for s in syms[bs:bs+BATCH_SIZE]
                   if s not in live_positions and s not in mv]
            scan_idx   = (scan_idx + 1) % n_bat
            scan_list  = list(dict.fromkeys(hot[:5] + mv[:20] + reg[:15]))[:BATCH_SIZE]
            if not scan_list:
                time.sleep(SLOT_FILL_INT)
                continue
            res = scan_batch(scan_list)
            if res:
                res.sort(key=lambda x: x[2], reverse=True)
                for r in res[:slots]:
                    if len(live_positions) >= MAX_POSITIONS:
                        break
                    sym, d, sc, sg, px, atr, regime, bias = r
                    live_open(sym, d, sc, sg, px, atr, regime, bias)
        except:
            pass
        time.sleep(SLOT_FILL_INT)


def t_rescan(syms: List[str]):
    while True:
        try:
            _rescan_q.get(timeout=5)
            time.sleep(0.05)
            slots = MAX_POSITIONS - len(live_positions)
            if slots <= 0 or ks_check()[0]:
                continue
            hot  = [s for s in _hot_syms if s not in live_positions]
            rest = [s for s in syms if s not in live_positions and s not in hot]
            res  = scan_batch((hot + rest)[:30])
            if res:
                res.sort(key=lambda x: x[2], reverse=True)
                for r in res[:slots]:
                    if len(live_positions) >= MAX_POSITIONS:
                        break
                    sym, d, sc, sg, px, atr, regime, bias = r
                    live_open(sym, d, sc, sg, px, atr, regime, bias)
        except:
            pass


def t_macro():
    while True:
        try:
            df_btc = ohlcv("BTCUSDT", Client.KLINE_INTERVAL_5MINUTE, 80)
            if df_btc is not None:
                regime, _, _ = MarketRegime.detect(df_btc)
                _macro["btc"] = regime
        except:
            pass
        time.sleep(10)

# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def run_bot():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  ✅ DRY RUN v20.1.1 — PROFIT-LOCK TRAILING + ANTI FEE LOSS          ║")
    print("║  🛡️  HardSL FIXED 0.40% | Trail ≥0.15% (breakeven lock)              ║")
    print("║  📈 Progressive Lock: 0.15%→BE | 0.25%→0.10% | 0.40%→0.15% ...     ║")
    print("║  ✅ Self-Learning ON | Reversed Logic OFF (bisa dinyalakan manual)  ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    try:
        valid = {s["symbol"]
                 for s in client.futures_exchange_info()["symbols"]
                 if s["status"] == "TRADING"}
        syms  = list(dict.fromkeys([s for s in SYMBOLS if s in valid]))
    except:
        syms = list(dict.fromkeys(SYMBOLS))

    print(f"  📋 {len(syms)} simbol aktif | "
          f"Monitor interval: {MONITOR_INT}s | "
          f"Trail phases: {len(TRAIL_PHASES)}")

    threading.Thread(target=t_monitor,               daemon=True).start()
    threading.Thread(target=t_slot_filler, args=(syms,), daemon=True).start()
    threading.Thread(target=t_rescan,      args=(syms,), daemon=True).start()
    threading.Thread(target=t_macro,                  daemon=True).start()

    time.sleep(2)
    tickers_all()

    cycle = 0
    while True:
        cycle += 1
        slots = MAX_POSITIONS - len(live_positions)
        ks_ok, ks_reason = ks_check()

        print(f"\n{'═'*64}")
        print(f"  #{cycle} {time.strftime('%H:%M:%S')} "
              f"BTC:{_macro['btc']} "
              f"({len(live_positions)}/{MAX_POSITIONS}) "
              f"PnL:{_stats['pnl']:+.4f}U")

        if ks_ok:
            print(f"  🚨 KS AKTIF: {ks_reason}")
        elif slots == 0:
            for sym, pos in list(live_positions.items()):
                if pos.get("_r"):
                    continue
                px = price_live(sym)
                if px == 0:
                    continue
                entry = pos["entry"]
                side  = pos["side"]
                gross = (px-entry)/entry if side=="LONG" else (entry-px)/entry
                if pos.get("trail_active"):
                    ts  = pos.get("trail_stop", 0)
                    gap = pos.get("trail_phase", 0)
                    print(f"  📌 {sym} {side} gross:{gross*100:+.3f}% "
                          f"TRAIL gap:{gap*100:.2f}% stop:{ts:.6g}")
                else:
                    print(f"  📌 {sym} {side} gross:{gross*100:+.3f}% "
                          f"HardSL:{pos['sl_price']:.6g}")
        else:
            print(f"  🔍 {slots} slot kosong — scanning...")

        if cycle % 30 == 0:
            print_full()

        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    run_bot()
