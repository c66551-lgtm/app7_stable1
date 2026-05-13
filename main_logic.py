import pandas as pd
import numpy as np

def detect_pivots(df, window=5):
    """
    偵測股價的轉折高點與低點 (Pivots)
    """
    pivots = []
    if len(df) < window * 2 + 1:
        return pivots

    for i in range(window, len(df) - window):
        is_high = True
        is_low = True
        for j in range(1, window + 1):
            if df['High'].iloc[i] < df['High'].iloc[i-j] or df['High'].iloc[i] < df['High'].iloc[i+j]:
                is_high = False
            if df['Low'].iloc[i] > df['Low'].iloc[i-j] or df['Low'].iloc[i] > df['Low'].iloc[i+j]:
                is_low = False
        
        if is_high:
            pivots.append((df.index[i], df['High'].iloc[i], 1)) # 1 代表高點
        elif is_low:
            pivots.append((df.index[i], df['Low'].iloc[i], -1)) # -1 代表低點
            
    return pivots

def detect_consolidation(df, window=20, threshold=0.08):
    """
    偵測是否有區間整理 (箱型)
    """
    recent = df.tail(window)
    h = recent['High'].max()
    l = recent['Low'].min()
    range_pct = (h - l) / (l + 1e-9)
    
    if range_pct < threshold:
        return {"status": True, "high": h, "low": l, "range": range_pct}
    return {"status": False, "high": h, "low": l, "range": range_pct}

def analyze_n_pattern(df, pivots):
    """
    分析 N 字突破結構
    """
    if len(pivots) < 3:
        return "數據不足"
    
    # 取得最近的三個轉折點 (低-高-低)
    p = [pt[1] for pt in pivots[-3:]]
    
    # N 字基礎結構：H1 > L1 且 L2 > L1
    if p[1] > p[0] and p[2] > p[0]:
        neckline = p[1] # 頸線為前波高點
        target = p[1] + (p[1] - p[0]) # 對等漲幅目標
        return {
            "neckline": neckline,
            "target": target,
            "base": p[2] # 防守點為次低點
        }
    return "尚未形成"

def calculate_volume_profile(df, bins=30):
    """
    計算成交量分佈，尋找主力成本區 (Value Area)
    """
    recent = df.tail(120)
    price_min = recent['Low'].min()
    price_max = recent['High'].max()
    
    if price_max == price_min:
        return None
        
    counts, bin_edges = np.histogram(recent['Close'], bins=bins, weights=recent['Volume'])
    
    # 找到成交量最大的價格帶
    max_idx = np.argmax(counts)
    vol_low = bin_edges[max_idx]
    vol_high = bin_edges[max_idx + 1]
    
    return (vol_low, vol_high)
