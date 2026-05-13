import pandas as pd
import numpy as np


def detect_pivots(df, window=5):
    """
    偵測股價轉折高點與低點
    回傳格式：
    [(日期index, 價格, 1或-1)]
    1 = 高點
    -1 = 低點
    """
    pivots = []

    if df is None or df.empty:
        return pivots

    if len(df) < window * 2 + 1:
        return pivots

    for i in range(window, len(df) - window):
        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]

        left_high = df["High"].iloc[i-window:i]
        right_high = df["High"].iloc[i+1:i+window+1]

        left_low = df["Low"].iloc[i-window:i]
        right_low = df["Low"].iloc[i+1:i+window+1]

        is_high = high >= left_high.max() and high >= right_high.max()
        is_low = low <= left_low.min() and low <= right_low.min()

        if is_high:
            pivots.append((df.index[i], float(high), 1))
        elif is_low:
            pivots.append((df.index[i], float(low), -1))

    return pivots


def detect_consolidation(df, window=20, threshold=0.08):
    """
    偵測箱型整理
    """
    if df is None or df.empty or len(df) < window:
        return {
            "status": False,
            "high": None,
            "low": None,
            "range": None
        }

    recent = df.tail(window)

    high = float(recent["High"].max())
    low = float(recent["Low"].min())

    if low <= 0:
        return {
            "status": False,
            "high": high,
            "low": low,
            "range": None
        }

    range_pct = (high - low) / low

    return {
        "status": range_pct < threshold,
        "high": high,
        "low": low,
        "range": float(range_pct)
    }


def analyze_n_pattern(df, pivots):
    """
    分析 N 字結構
    必須符合：低點 -> 高點 -> 較高低點
    回傳 dict 或字串
    """
    if pivots is None or len(pivots) < 3:
        return "數據不足"

    # 從最近 pivots 裡找最後一組 低-高-低
    for i in range(len(pivots) - 3, -1, -1):
        p1, p2, p3 = pivots[i], pivots[i + 1], pivots[i + 2]

        is_lhl = p1[2] == -1 and p2[2] == 1 and p3[2] == -1

        if not is_lhl:
            continue

        low1 = float(p1[1])
        high1 = float(p2[1])
        low2 = float(p3[1])

        # N 字：第二低點要高於第一低點
        if high1 > low1 and low2 > low1:
            neckline = high1
            target = high1 + (high1 - low1)

            return {
                "neckline": float(neckline),
                "target": float(target),
                "base": float(low2)
            }

    return "尚未形成"


def calculate_volume_profile(df, bins=30):
    """
    計算主力成本區
    回傳格式：
    (成本區下緣, 成本區上緣)
    """
    if df is None or df.empty:
        return None

    recent = df.tail(120).copy()

    if len(recent) < 30:
        return None

    price_min = float(recent["Low"].min())
    price_max = float(recent["High"].max())

    if price_max <= price_min:
        return None

    close = recent["Close"].astype(float)
    volume = recent["Volume"].astype(float)

    if volume.sum() <= 0:
        return None

    counts, bin_edges = np.histogram(
        close,
        bins=bins,
        range=(price_min, price_max),
        weights=volume
    )

    if len(counts) == 0 or counts.max() <= 0:
        return None

    max_idx = int(np.argmax(counts))

    vol_low = float(bin_edges[max_idx])
    vol_high = float(bin_edges[max_idx + 1])

    return vol_low, vol_high
