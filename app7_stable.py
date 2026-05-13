import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import numpy as np
import os
import json
import joblib
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier

try:
    from main_logic import (
        detect_pivots,
        detect_consolidation,
        analyze_n_pattern,
        calculate_volume_profile,
    )
except ImportError as e:
    st.error(f"❌ 找不到 main_logic.py：{e}")
    st.stop()

st.set_page_config(layout="wide", page_title="AI 量化診斷 Stable V7")

st.markdown("""
<style>

/* ===== 全站文字：不要影響 sidebar input/button ===== */
html, body, [data-testid="stAppViewContainer"] {
    font-size: 20px !important;
}

p, li {
    font-size: 20px !important;
    line-height: 1.8 !important;
}

/* ===== Sidebar 縮小：股票代號框與啟動按鈕 ===== */
section[data-testid="stSidebar"] label {
    font-size: 16px !important;
    line-height: 1.3 !important;
}

section[data-testid="stSidebar"] input {
    font-size: 16px !important;
    height: 36px !important;
    padding: 4px 8px !important;
}

section[data-testid="stSidebar"] button {
    font-size: 16px !important;
    padding: 0.35rem 0.7rem !important;
    min-height: 36px !important;
}

/* ===== 標題 ===== */
h1 { font-size: 48px !important; font-weight: 900 !important; }
h2 { font-size: 36px !important; font-weight: 800 !important; }
h3 { font-size: 30px !important; font-weight: 800 !important; }
h4 { font-size: 26px !important; font-weight: 800 !important; }

/* ===== warning/info ===== */
[data-testid="stAlert"] {
    font-size: 22px !important;
    line-height: 1.8 !important;
}

[data-testid="stInfo"] * {
    font-size: 22px !important;
    line-height: 1.85 !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    font-size:64px;
    font-weight:900;
    margin-bottom:25px;
    line-height:1.2;
">
📈 AI 股票量化診斷 - Stable V7
</div>
""", unsafe_allow_html=True)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


@st.cache_data(ttl=3600)
def load_stock_data(ticker):

    try:
        df = yf.Ticker(ticker).history(
            period="3y",
            auto_adjust=True
        )

        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)

        return df

    except Exception as e:
        st.error(f"下載失敗：{e}")
        return pd.DataFrame()


def normalize_ticker(raw_symbol):
    raw = raw_symbol.strip().upper()

    if not raw.isdigit():
        return raw

    code = raw.zfill(4)

    candidates = [
        f"{code}.TW",
        f"{code}.TWO",
    ]

    for tk in candidates:
        try:
            test = yf.download(
                tk,
                period="5d",
                auto_adjust=True,
                progress=False
            )

            if test is not None and not test.empty:
                return tk
        except Exception:
            pass

    return f"{code}.TWO"


def get_stock_name(ticker):
    try:
        info = yf.Ticker(ticker).info or {}
        return info.get("longName") or info.get("shortName") or ticker
    except:
        return ticker


def get_financials(ticker, pe_multiple=20):
    try:
        info = yf.Ticker(ticker).info or {}
        eps = info.get("forwardEps") or info.get("trailingEps")

        if eps is not None and float(eps) > 0:
            return float(eps), float(eps) * pe_multiple

        return None, None
    except:
        return None, None


def analyze_elliott_waves(pivots):
    if len(pivots) < 5:
        return "數據不足以進行波浪推算"

    p = [pt[1] for pt in pivots[-5:]]
    w1_len = abs(p[1] - p[0])
    w3_len = abs(p[3] - p[2])

    if p[4] > p[2] > p[0] and p[3] > p[1]:
        if w3_len >= w1_len * 1.618:
            return "🌊 強勢主升段：第 3 浪擴張中"
        return "🌊 衝刺階段：疑似處於第 5 浪階段"

    if p[4] < p[2] and p[3] < p[1]:
        return "🌊 修正階段：疑似處於 C 浪調整"

    return "🌊 震盪階段：波浪結構重組中"


def analyze_market_structure(pivots):
    if len(pivots) < 4:
        return "結構重組中", 0

    highs = [p[1] for p in pivots if p[2] == 1][-2:]
    lows = [p[1] for p in pivots if p[2] == -1][-2:]

    if len(highs) < 2 or len(lows) < 2:
        return "結構不明", 0

    is_hh = highs[-1] > highs[-2]
    is_hl = lows[-1] > lows[-2]
    is_lh = highs[-1] < highs[-2]
    is_ll = lows[-1] < lows[-2]

    if is_hh and is_hl:
        return "多頭結構 (HH+HL)", 20

    if is_lh and is_ll:
        return "空頭結構 (LH+LL)", -20

    if is_lh and is_hl:
        return "收斂三角形", 5

    return "橫盤 / 擴張", 0


def classify_trend_regime(df, curr_p, fib, vol_zone, ai_prob, pivots):
    ma5 = df["Close"].rolling(5).mean().iloc[-1]
    ma20 = df["Close"].rolling(20).mean().iloc[-1]
    ma60 = df["Close"].rolling(60).mean().iloc[-1]
    ma120 = df["Close"].rolling(120).mean().iloc[-1]

    score = 0

    if curr_p > ma5 > ma20 > ma60:
        score += 50
    elif curr_p > ma20 > ma60:
        score += 35
    elif curr_p > ma60:
        score += 15

    if ma60 > ma120:
        score += 10

    struct_desc, struct_score = analyze_market_structure(pivots)
    score += struct_score

    if vol_zone is not None and curr_p > vol_zone[1]:
        score += 10

    if fib is not None and curr_p > fib["fib_382"]:
        score += 10
    elif fib is not None and curr_p > fib["fib_500"]:
        score += 5

    score += ai_prob * 20

    bear_filter = curr_p < ma60 and ma20 < ma60
    if bear_filter:
        score -= 25

    if score >= 80:
        res = {
            "status": "🔥 強勢多頭",
            "bias": "極度看多",
            "mode": "積極進場、突破加碼",
        }
    elif 65 <= score < 80:
        if curr_p < ma5:
            res = {
                "status": "🍃 健康修正",
                "bias": "偏多回測",
                "mode": "尋求支撐分批佈局",
            }
        else:
            res = {
                "status": "📈 溫和多頭",
                "bias": "看多",
                "mode": "不追高，回測買進",
            }
    elif 45 <= score < 65:
        res = {
            "status": "⚖️ 高檔震盪",
            "bias": "中性偏多",
            "mode": "區間低買高賣",
        }
    else:
        res = {
            "status": "❄️ 空頭 / 轉弱",
            "bias": "看空",
            "mode": "觀望為主，嚴格止損",
        }

    res.update(
        {
            "score": int(score),
            "struct": struct_desc,
            "bear": "啟動" if bear_filter else "關閉",
        }
    )

    return res


def detect_trend_reversal(df, ai_prob):
    ma5 = df["Close"].rolling(5).mean()
    ma20 = df["Close"].rolling(20).mean()

    recent_close = df["Close"].iloc[-1]
    ma_gap = (ma5.iloc[-1] - ma20.iloc[-1]) / (ma20.iloc[-1] + 1e-9)

    vol_sma20 = df["Volume"].rolling(20).mean().iloc[-1]
    volume_ratio = df["Volume"].iloc[-1] / vol_sma20 if vol_sma20 > 0 else 1

    momentum = df["Close"].pct_change(5).iloc[-1]

    reversal_score = 0

    if momentum < 0:
        reversal_score += 25
    if ma_gap < 0.01:
        reversal_score += 20
    if volume_ratio < 0.85:
        reversal_score += 15
    if ai_prob < 0.55:
        reversal_score += 25

    if recent_close > ma5.iloc[-1] and volume_ratio > 1.2 and ai_prob > 0.60:
        return {
            "status": "🔄 空頭反轉疑似形成",
            "message": "空方結構可能結束，需觀察是否站穩 MA20。",
        }

    if reversal_score >= 70:
        return {
            "status": "⚠ 多頭趨勢衰退",
            "message": "趨勢動能明顯減弱，需防範高檔修正。",
        }

    if reversal_score >= 45:
        return {
            "status": "🍃 多頭動能減弱",
            "message": "趨勢仍偏多，但追價風險開始提高。",
        }

    return {
        "status": "✅ 趨勢健康",
        "message": "目前趨勢結構仍穩定。",
    }


def detect_institutional_behavior(df):
    close = df["Close"]
    volume = df["Volume"]

    recent_range = (close.tail(20).max() - close.tail(20).min()) / (
        close.iloc[-1] + 1e-9
    )

    vol_mean_5 = volume.tail(5).mean()
    vol_mean_20 = volume.tail(20).mean()
    volume_trend = vol_mean_5 / vol_mean_20 if vol_mean_20 > 0 else 1

    obv = pd.Series(
        np.where(close > close.shift(1), volume, -volume),
        index=df.index,
    ).cumsum()

    obv_trend = obv.iloc[-5:].mean() - obv.iloc[-20:].mean()

    if recent_range < 0.08 and obv_trend > 0 and volume_trend < 1:
        return {
            "status": "🏦 主力吸籌",
            "message": "價格整理但資金流入增加，疑似主力低調吸籌。",
        }

    if recent_range > 0.12 and obv_trend < 0 and volume_trend > 1.3:
        return {
            "status": "⚠ 主力出貨",
            "message": "高檔爆量但資金流出，需留意轉弱風險。",
        }

    return {
        "status": "📊 籌碼中性",
        "message": "目前未觀察到明顯吸籌或出貨跡象。",
    }
def calculate_dynamic_stoploss(
    curr_p,
    atr,
    fib,
    vol_zone,
    n_pattern
):
    atr_stop = curr_p - atr * 1.5

    fib_stop = (
        fib["fib_618"]
        if fib is not None
        else curr_p * 0.92
    )

    vol_stop = (
        vol_zone[0]
        if vol_zone is not None
        else curr_p * 0.93
    )

    n_stop = (
        n_pattern.get("base")
        if isinstance(n_pattern, dict)
        else curr_p * 0.90
    )

    stop_price = max(
        atr_stop,
        fib_stop,
        vol_stop,
        n_stop
    )

    stop_pct = (
        (stop_price / curr_p) - 1
    ) * 100

    if stop_pct > -4:
        level = "🟢 緊密停損"
    elif stop_pct > -8:
        level = "🟡 標準停損"
    else:
        level = "🔴 寬幅停損"

    return {
        "price": stop_price,
        "pct": stop_pct,
        "level": level
    }

def detect_fake_breakout(df, curr_p):
    recent_high = df["High"].tail(20).max()

    recent_vol = df["Volume"].iloc[-1]
    avg_vol = df["Volume"].tail(20).mean()

    candle_body = abs(
        df["Close"].iloc[-1]
        - df["Open"].iloc[-1]
    )

    upper_shadow = (
        df["High"].iloc[-1]
        - max(
            df["Close"].iloc[-1],
            df["Open"].iloc[-1]
        )
    )

    breakout = curr_p >= recent_high * 0.995

    weak_volume = recent_vol < avg_vol * 1.1

    long_upper_shadow = (
        upper_shadow > candle_body * 1.2
    )

    if breakout and weak_volume:
        return {
            "status": "⚠ 疑似假突破",
            "message": "突破前高但量能不足，追價風險偏高。"
        }

    if breakout and long_upper_shadow:
        return {
            "status": "⚠ 上影線過長",
            "message": "高檔賣壓開始出現，需觀察是否跌回壓力區。"
        }

    return {
        "status": "✅ 突破結構正常",
        "message": "量價結構尚未出現明顯假突破訊號。"
    }

def evaluate_support_strength(curr_p, fib, vol_zone, ma20):
    score = 0
    supports = []

    if fib is not None:
        fib382 = fib["fib_382"]
        fib500 = fib["fib_500"]

        # 站上 Fib 38.2，代表強勢支撐在下方
        if curr_p >= fib382:
            score += 25
            supports.append("Fib 38.2 下方支撐")
        elif abs(curr_p - fib382) / curr_p < 0.05:
            score += 15
            supports.append("接近 Fib 38.2")

        if curr_p >= fib500:
            score += 15
            supports.append("Fib 50 防守")

    if vol_zone is not None:
        # 在主力成本上方，也應該算支撐，不要只算「在區間內」
        if curr_p >= vol_zone[1]:
            score += 30
            supports.append("站上主力成本")
        elif vol_zone[0] <= curr_p <= vol_zone[1]:
            score += 25
            supports.append("主力成本區內")
        elif abs(curr_p - vol_zone[1]) / curr_p < 0.05:
            score += 15
            supports.append("接近主力成本")

    ma_gap = abs(curr_p - ma20) / curr_p

    if curr_p >= ma20:
        score += 20
        supports.append("站上 MA20")
    elif ma_gap < 0.05:
        score += 10
        supports.append("接近 MA20")

    score = min(score, 100)

    if score >= 70:
        level = "🟢 強支撐"
    elif score >= 40:
        level = "🟡 中性支撐"
    else:
        level = "🔴 弱支撐"

    return {
        "score": score,
        "level": level,
        "supports": supports
    }
def calculate_ai_grade(ai_prob, win_rate, trend_score, support_score):
    total = (
        ai_prob * 35 +
        win_rate * 30 +
        trend_score * 0.25 +
        support_score * 0.10
    )

    if total >= 85:
        return "S級"
    elif total >= 70:
        return "A級"
    elif total >= 55:
        return "B級"
    else:
        return "C級"

def build_ai_features(df):
    data = df.copy()

    data["return_1d"] = data["Close"].pct_change()
    data["ma5_gap"] = (
        data["Close"] - data["Close"].rolling(5).mean()
    ) / (data["Close"].rolling(5).mean() + 1e-9)
    data["ma20_gap"] = (
        data["Close"] - data["Close"].rolling(20).mean()
    ) / (data["Close"].rolling(20).mean() + 1e-9)
    data["ma60_gap"] = (
        data["Close"] - data["Close"].rolling(60).mean()
    ) / (data["Close"].rolling(60).mean() + 1e-9)

    data["ema_spread"] = (
        data["Close"].ewm(span=12).mean()
        - data["Close"].ewm(span=26).mean()
    ) / (data["Close"] + 1e-9)

    data["slope_20"] = data["Close"].rolling(20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] / x[-1]
        if len(x) == 20
        else np.nan,
        raw=True,
    )

    tr = pd.concat(
        [
            data["High"] - data["Low"],
            abs(data["High"] - data["Close"].shift(1)),
            abs(data["Low"] - data["Close"].shift(1)),
        ],
        axis=1,
    ).max(axis=1)

    data["atr_pct"] = tr.rolling(14).mean() / (data["Close"] + 1e-9)
    data["realized_vol"] = data["Close"].pct_change().rolling(20).std()
    data["volume_ratio"] = data["Volume"] / (
        data["Volume"].rolling(20).mean() + 1e-9
    )
    data["volume_delta"] = data["Volume"].pct_change()

    data["obv_gap"] = pd.Series(
        np.where(data["Close"] > data["Close"].shift(1), 1, -1)
        * data["Volume"],
        index=data.index,
    ).cumsum().pct_change(20)

    tp = (data["High"] + data["Low"] + data["Close"]) / 3

    data["vwap_gap"] = (
        data["Close"]
        - (tp * data["Volume"]).rolling(20).sum()
        / (data["Volume"].rolling(20).sum() + 1e-9)
    ) / (data["Close"] + 1e-9)

    data["roc_10"] = data["Close"].pct_change(10)

    data["stochastic"] = (
        data["Close"] - data["Low"].rolling(14).min()
    ) / (
        data["High"].rolling(14).max()
        - data["Low"].rolling(14).min()
        + 1e-9
    )

    data["breakout_strength"] = (
        data["Close"] / (data["High"].rolling(20).max().shift(1) + 1e-9)
        - 1
    )

    data["consolidation_tightness"] = (
        data["High"].rolling(20).max()
        - data["Low"].rolling(20).min()
    ) / (data["Close"] + 1e-9)

    data["target"] = (
        data["Close"].shift(-5) / data["Close"] - 1 > 0.03
    ).astype(int)

    features = [
        "return_1d",
        "ma5_gap",
        "ma20_gap",
        "ma60_gap",
        "ema_spread",
        "slope_20",
        "atr_pct",
        "realized_vol",
        "volume_ratio",
        "volume_delta",
        "obv_gap",
        "vwap_gap",
        "roc_10",
        "stochastic",
        "breakout_strength",
        "consolidation_tightness",
    ]

    return data.replace([np.inf, -np.inf], np.nan).dropna(), features


def get_model_paths(ticker):
    safe = ticker.replace(".", "_")
    return {
        "model": os.path.join(MODEL_DIR, f"{safe}_stable_rf.pkl"),
        "meta": os.path.join(MODEL_DIR, f"{safe}_stable_meta.json"),
    }


def need_retrain(ticker, latest_date):
    paths = get_model_paths(ticker)

    if not os.path.exists(paths["model"]) or not os.path.exists(paths["meta"]):
        return True

    try:
        with open(paths["meta"], "r", encoding="utf-8") as f:
            meta = json.load(f)

        return str(latest_date.date()) > meta.get("latest_data_date", "")
    except:
        return True


def walk_forward_train(df):
    data, features = build_ai_features(df)

    if len(data) < 300:
        return None

    train_days = 240
    test_days = 60

    all_probs = []
    all_index = []

    for start in range(0, len(data) - train_days - test_days, test_days):
        train = data.iloc[start : start + train_days]
        test = data.iloc[start + train_days : start + train_days + test_days]

        if train["target"].nunique() < 2:
            continue

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=7,
            random_state=42,
        )

        model.fit(train[features], train["target"])

        all_probs.extend(model.predict_proba(test[features])[:, 1])
        all_index.extend(test.index)

    if not all_probs:
        return None

    win_rate = (
        (np.array(all_probs) > 0.55).astype(int)
        == data.loc[all_index, "target"].values
    ).mean()

    final_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=7,
        random_state=42,
    )

    final_model.fit(data[features], data["target"])

    return final_model, features, float(win_rate)


def run_advanced_backtest(df, ticker):
    latest_date = df.index[-1]
    paths = get_model_paths(ticker)

    if need_retrain(ticker, latest_date):
        trained = walk_forward_train(df)

        if trained is None:
            return None

        model, features, win_rate = trained

        joblib.dump({"model": model, "features": features}, paths["model"])

        with open(paths["meta"], "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ticker": ticker,
                    "latest_data_date": str(latest_date.date()),
                    "walk_forward_win_rate": win_rate,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    try:
        saved = joblib.load(paths["model"])

        with open(paths["meta"], "r", encoding="utf-8") as f:
            meta = json.load(f)

        data_now, _ = build_ai_features(df)

        if data_now.empty:
            return None

        last_prob = float(
            saved["model"].predict_proba(
                data_now[saved["features"]].iloc[[-1]]
            )[:, 1][0]
        )

        return {
            "last_prob": last_prob,
            "win_rate": float(meta.get("walk_forward_win_rate", 0)),
        }

    except Exception as e:
        st.error(f"AI 模型讀取失敗：{e}")
        return None


def calculate_fibonacci_levels(df):
    recent = df.tail(120)

    sh = float(recent["High"].max())
    sl = float(recent["Low"].min())
    diff = sh - sl

    if diff <= 0:
        return None

    return {
        "fib_382": sh - diff * 0.382,
        "fib_500": sh - diff * 0.5,
        "fib_618": sh - diff * 0.618,
    }


def generate_v10_plot(df, ticker, stock_name, pivots, vol_zone, fib):
    plot_df = df.tail(120).copy()

    mc = mpf.make_marketcolors(
        up="red",
        down="green",
        edge="inherit",
        wick="inherit",
        volume="inherit",
    )

    ap = [
        mpf.make_addplot(
            plot_df["Close"].rolling(5).mean(),
            color="royalblue",
            width=0.8,
        ),
        mpf.make_addplot(
            plot_df["Close"].rolling(20).mean(),
            color="darkorange",
            width=1.2,
        ),
    ]

    fig, axlist = mpf.plot(
        plot_df,
        type="candle",
        style=mpf.make_mpf_style(marketcolors=mc, gridstyle="--"),
        volume=False,
        addplot=ap,
        figsize=(16, 10),
        returnfig=True,
    )

    ax = axlist[0]
    tail_offset = len(df) - len(plot_df)

    ax_vol = ax.twinx()
    v_cols = np.where(
        plot_df["Close"] >= plot_df["Close"].shift(1),
        "red",
        "green",
    )

    ax_vol.bar(
        range(len(plot_df)),
        plot_df["Volume"],
        color=v_cols,
        alpha=0.15,
    )
    ax_vol.set_ylim(0, plot_df["Volume"].max() * 4)
    ax_vol.set_yticks([])

    pts = []

    for p in pivots:
        x = (
            p[0] - tail_offset
            if isinstance(p[0], (int, np.integer))
            else plot_df.index.get_loc(p[0])
            if p[0] in plot_df.index
            else -1
        )

        if 0 <= x < len(plot_df):
            pts.append((x, p[1]))
            ax.text(
                x,
                p[1],
                f"{p[1]:.1f}",
                color="blue",
                fontsize=9,
                ha="center",
                va="bottom",
                bbox=dict(facecolor="white", alpha=0.7),
            )

    if len(pts) >= 2:
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            color="blue",
            lw=2,
            ls="--",
            alpha=0.7,
        )

    if fib is not None:
        for lbl, price in [
            ("38.2%", fib["fib_382"]),
            ("50.0%", fib["fib_500"]),
            ("61.8%", fib["fib_618"]),
        ]:
            ax.axhline(price, ls="--", alpha=0.4, color="gray")
            ax.text(
                len(plot_df) * 0.9,
                price,
                f"{lbl}: {price:.1f}",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7),
            )

    if vol_zone is not None:
        ax.axhspan(vol_zone[0], vol_zone[1], color="yellow", alpha=0.08)

    ax.set_title(f"{ticker} {stock_name} - AI Stable V7", fontsize=18)

    return fig

def metric_card(title, value, subtext="", accent="#2f80ff", icon=""):
    html = f"""
    <div style="
        width:100%;
        height:190px;
        padding:24px 26px;
        border-radius:18px;
        background:linear-gradient(145deg,rgba(255,255,255,0.95),rgba(245,245,245,0.92));
        border:1px solid {accent};
        box-shadow:0 0 22px rgba(0,0,0,0.28);
        box-sizing:border-box;
        font-family:Arial, sans-serif;
        color:#222;
    ">
        <div style="
            font-size:22px;
            font-weight:800;
            margin-bottom:24px;
            white-space:nowrap;
        ">
            {icon} {title}
        </div>

        <div style="
            font-size:30px;
            font-weight:950;
            color:{accent};
            line-height:1.1;
            white-space:nowrap;
        ">
            {value}
        </div>

        <div style="
            margin-top:22px;
            padding:9px 14px;
            border-radius:10px;
            background:rgba(0,0,0,0.05);
            border:1px solid rgba(255,255,255,0.12);
            font-size:21px;
            font-weight:800;
            color:#333;
            white-space:normal;
            word-break:break-word;
        ">
            {subtext}
        </div>
    </div>
    """

    components.html(html, height=210)

def neon_panel(title, icon, content_html, accent="#2f80ff", height=500):
    html = f"""
    <div style="
        width:100%;
        padding:26px 30px;
        margin-top:18px;
        margin-bottom:18px;
        border-radius:20px;
        background:
            background:linear-gradient(145deg,rgba(255,255,255,0.96),rgba(245,245,245,0.93));
        border:1px solid rgba(255,255,255,0.14);
        box-shadow:0 0 26px rgba(0,0,0,0.32), inset 0 0 18px rgba(255,255,255,0.025);
        box-sizing:border-box;
        color:#222;
    ">
        <div style="
            font-size:28px;
            font-weight:900;
            margin-bottom:22px;
           color:#222;
        ">
            <span style="color:{accent}; font-size:30px;">{icon}</span>
            {title}
        </div>

        <div style="
            font-size:21px;
            line-height:1.9;
            font-weight:650;
        ">
            {content_html}
        </div>
    </div>
    """

    components.html(html, height=height, scrolling=False)

def mini_price_box(title, value, note="", accent="#2f80ff"):
    return f"""
    <div style="
        padding:18px 20px;
        border-radius:16px;
        background:rgba(255,255,255,0.045);
        border:1px solid {accent}66;
        min-height:105px;
        box-sizing:border-box;
    ">
        <div style="font-size:18px; opacity:0.82; font-weight:800;">
            {title}
        </div>
        <div style="font-size:30px; font-weight:950; color:{accent}; margin-top:8px;">
            {value}
        </div>
        <div style="font-size:17px; opacity:0.78; margin-top:6px;">
            {note}
        </div>
    </div>
    """

with st.sidebar:
    symbol = st.text_input("股票代碼")
    run = st.button("🚀 啟動診斷")


if run:
    tk = normalize_ticker(symbol)
    df = load_stock_data(tk)

    if df.empty:
        st.error("❌ 查無資料")
        st.stop()

    curr_p = float(df["Close"].iloc[-1])

    eps, fin_target = get_financials(tk, 20)

    bt = run_advanced_backtest(df, tk)

    if bt is None:
        st.error("❌ AI 核心訓練失敗或資料不足")
        st.stop()

    pivots = detect_pivots(df)
    vol_zone = calculate_volume_profile(df)
    fib = calculate_fibonacci_levels(df)

    n_pattern = analyze_n_pattern(df, pivots)
    box = detect_consolidation(df)
    wave_info = analyze_elliott_waves(pivots)

    regime = classify_trend_regime(
        df,
        curr_p,
        fib,
        vol_zone,
        bt["last_prob"],
        pivots,
    )

    reversal = detect_trend_reversal(df, bt["last_prob"])
    institution = detect_institutional_behavior(df)

    fake_breakout = detect_fake_breakout(
        df,
        curr_p
    )

    support_eval = evaluate_support_strength(
        curr_p,
        fib,
        vol_zone,
        df["Close"].rolling(20).mean().iloc[-1]
    )

    ai_grade = calculate_ai_grade(
        ai_prob=bt["last_prob"],
        win_rate=bt["win_rate"],
        trend_score=regime["score"],
        support_score=support_eval["score"]
    )

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            abs(df["High"] - df["Close"].shift(1)),
            abs(df["Low"] - df["Close"].shift(1)),
        ],
        axis=1,
    ).max(axis=1)

    atr = float(tr.rolling(14).mean().iloc[-1])
    atr_pct = atr / curr_p

    ai_prob = bt["last_prob"]
    win_rate = bt["win_rate"]
    recent_high = float(df["High"].tail(20).max())

    base_gain = atr_pct * 1.8
    ai_bonus = max(0, ai_prob - 0.5) * 0.18
    win_bonus = max(0, win_rate - 0.5) * 0.12

    target_gain_1 = min(
        max(base_gain + ai_bonus * 0.4 + win_bonus * 0.3, 0.025),
        0.08,
    )

    target_gain_2 = min(
        max(base_gain * 1.6 + ai_bonus * 0.7 + win_bonus * 0.5, 0.04),
        0.13,
    )

    target_gain_3 = min(
        max(base_gain * 2.5 + ai_bonus + win_bonus, 0.07),
        0.22,
    )

    t1 = max(curr_p * (1 + target_gain_1), recent_high)
    t2 = curr_p * (1 + target_gain_2)
    t3 = curr_p * (1 + target_gain_3)

    short_buy = curr_p

    dynamic_stop = calculate_dynamic_stoploss(
        curr_p=curr_p,
        atr=atr,
        fib=fib,
        vol_zone=vol_zone,
        n_pattern=n_pattern
    )

    n_buy = n_pattern.get("neckline", None) if isinstance(n_pattern, dict) else None
    n_target = n_pattern.get("target", None) if isinstance(n_pattern, dict) else None
    n_base = n_pattern.get("base", None) if isinstance(n_pattern, dict) else None

    swing_buy_1 = fib["fib_382"] if fib is not None else curr_p * 0.97
    swing_buy_2 = fib["fib_500"] if fib is not None else curr_p * 0.94
    swing_stop = n_base if n_base else (
        fib["fib_618"] if fib is not None else curr_p * 0.90
    )

    add_price = recent_high
    swing_target_1 = n_target if n_target else t2
    swing_target_2 = max(t3, recent_high * 1.08)

    st.header(f"🔍 {tk} {get_stock_name(tk)} AI 穩定版診斷報告")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        metric_card("目前股價", f"${curr_p:.1f}", "即時收盤價", "#2ecc71", "💵")

    with m2:
        metric_card("趨勢分數", f"{regime['score']}分", regime["status"], "#2f80ff", "📈")

    with m3:
        metric_card("市場結構", regime["struct"], "結構判定", "#bb6bd9", "🧩")

    with m4:
        metric_card("AI 噴發率", f"{bt['last_prob']:.1%}", ai_grade, "#f2a900", "🔥")

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    st.pyplot(generate_v10_plot(df, tk, get_stock_name(tk), pivots, vol_zone, fib))

    st.subheader("📝 AI 趨勢導引與深度點評")
    st.warning(
        f"## {regime['status']} ｜ {ai_grade}\n\n"
        f"**市場偏向：{regime['bias']}**\n\n"
        f"**操作邏輯：{regime['mode']}**"
    )

    analysis_html = f"""
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:18px;">
        <div>
            🧩 <b>結構狀態：</b>{regime['struct']}<br>
            ❄️ <b>空頭過濾：</b>{regime['bear']}<br>
            🔄 <b>趨勢轉折：</b>{reversal['status']}<br>
            🏦 <b>主力行為：</b>{institution['status']}<br>
            ⚠️ <b>突破風險：</b>{fake_breakout['status']}<br>
        </div>
        <div>
            📌 <b>主力成本：</b>{f'{vol_zone[0]:.1f} ～ {vol_zone[1]:.1f}' if vol_zone is not None else '數據不足'}<br>
            📐 <b>Fibonacci：</b>{f"38.2%：{fib['fib_382']:.1f}｜50%：{fib['fib_500']:.1f}｜61.8%：{fib['fib_618']:.1f}" if fib is not None else "數據不足"}<br>
            🌊 <b>市場波浪：</b>{wave_info}<br>
            🛡️ <b>支撐強度：</b>{support_eval['level']}，{support_eval['score']} 分<br>
            💰 <b>20x 本益比估值：</b>{f"${fin_target:.1f}" if fin_target else "數據不足"}<br>
        </div>
    </div>

    <div style="
        margin-top:18px;
        padding:16px 18px;
        border-radius:14px;
        background:rgba(47,128,255,0.18);
        border:1px solid rgba(47,128,255,0.35);
    ">
        <b>AI 解讀：</b>{reversal['message']}<br>
        <b>籌碼分析：</b>{institution['message']}<br>
        <b>突破分析：</b>{fake_breakout['message']}<br>
        <b>有效支撐來源：</b>{', '.join(support_eval['supports']) if support_eval['supports'] else '無明顯支撐'}
    </div>
    """

    neon_panel(
        "形態、轉折與籌碼分析",
        "📊",
        analysis_html,
        "#2f80ff",
        height=750
    )
    trade_html = f"""
    <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:16px;">
        {mini_price_box("短線買入", f"{short_buy:.1f}", "目前參考價", "#2ecc71")}
        {mini_price_box("動態停損", f"{dynamic_stop['price']:.1f}", f"{dynamic_stop['pct']:.1f}%｜{dynamic_stop['level']}", "#eb5757")}
        {mini_price_box("第一目標", f"{t1:.1f}", f"+{(t1 / curr_p - 1) * 100:.1f}%", "#f2a900")}
        {mini_price_box("第二目標", f"{t2:.1f}", f"+{(t2 / curr_p - 1) * 100:.1f}%", "#f2994a")}
        {mini_price_box("強勢目標", f"{t3:.1f}", f"+{(t3 / curr_p - 1) * 100:.1f}%", "#bb6bd9")}
        {mini_price_box("波段加碼", f"{add_price:.1f}", "突破前高且回測不破", "#2f80ff")}
    </div>

    <div style="
        margin-top:18px;
        padding:16px 18px;
        border-radius:14px;
        background:rgba(242,169,0,0.10);
        border:1px solid rgba(242,169,0,0.35);
    ">
        <b>回檔買點：</b>{swing_buy_1:.1f} / {swing_buy_2:.1f}<br>
        <b>N 字突破買點：</b>{f'{n_buy:.1f}' if isinstance(n_buy, (int, float, np.floating)) else '尚未形成明確頸線'}<br>
        <b>波段第一目標價：</b>{f'{swing_target_1:.1f}' if isinstance(swing_target_1, (int, float, np.floating)) else '尚未形成'}<br>
        <b>波段強勢目標價：</b>{swing_target_2:.1f}<br>
        <b>波段停損價：</b>{min(curr_p * 0.94, swing_stop if swing_stop is not None else curr_p):.1f}<br>
        <b>操作條件：</b>帶量突破、回測不破，且 AI 噴發率維持強勢。
    </div>
    """

    neon_panel(
        "AI 短線 / 波段操作價位建議",
        "🎯",
        trade_html,
        "#f2a900",
        height=820
    )

    st.info(f"""
### 📊 形態、轉折與籌碼分析
- **結構狀態**：{regime['struct']}
- **空頭過濾**：{regime['bear']}
- **趨勢轉折偵測**：{reversal['status']}
- **轉折分析**：{reversal['message']}
- **主力行為**：{institution['status']}
- **籌碼分析**：{institution['message']}
- **N 字結構**：目標 {n_pattern['target'] if isinstance(n_pattern, dict) else '無'}｜頸線 {n_pattern.get('neckline', '無') if isinstance(n_pattern, dict) else '無'}｜防守 {n_pattern.get('base', '無') if isinstance(n_pattern, dict) else '無'}
- **主力成本**：{f'{vol_zone[0]:.1f} ～ {vol_zone[1]:.1f}' if vol_zone is not None else '數據不足'}
- **Fibonacci**：{f"38.2%：{fib['fib_382']:.1f}｜50%：{fib['fib_500']:.1f}｜61.8%：{fib['fib_618']:.1f}" if fib is not None else "數據不足"}
- **市場波浪**：{wave_info}
- **20x 本益比估值目標**：{f"${fin_target:.1f}" if fin_target else "數據不足"}
- **突破風險**：{fake_breakout['status']}
- **突破分析**：{fake_breakout['message']}

- **支撐強度**：{support_eval['level']}
- **支撐評分**：{support_eval['score']} 分
- **有效支撐來源**：{', '.join(support_eval['supports']) if support_eval['supports'] else '無明顯支撐'}
---

#### 🗳️ 模型投票系統
- **均線排列**：{'🔥 多頭排列' if curr_p > df["Close"].rolling(20).mean().iloc[-1] else '❄️ 尚未轉強'}
- **結構判定**：{regime['struct']}
- **AI 噴發率**：{'✅ 具備攻擊力' if bt['last_prob'] > 0.55 else '❌ 動能不足'}
- **回測勝率**：{bt['win_rate']:.1%}

---

#### 🎯 AI 短線 / 波段操作價位建議

##### ⚡ 一筆小資金進出建議(短線)
- **短線參考買入價**：**{short_buy:.1f}**
- **動態停損價**：**{dynamic_stop['price']:.1f}**
- **預估停損幅度**：**{dynamic_stop['pct']:.1f}%**
- **停損等級**：{dynamic_stop['level']}
- **第一目標價**：**{t1:.1f}**，預估漲幅 **{(t1 / curr_p - 1) * 100:.1f}%**
- **第二目標價**：**{t2:.1f}**，預估漲幅 **{(t2 / curr_p - 1) * 100:.1f}%**
- **強勢目標價**：**{t3:.1f}**，預估漲幅 **{(t3 / curr_p - 1) * 100:.1f}%**

##### 📈 回檔做波段建議
- **第一波段買點**：{swing_buy_1:.1f}，參考 Fib 38.2%
- **第二波段買點**：{swing_buy_2:.1f}，參考 Fib 50%
- **N 字突破買點**：{f'{n_buy:.1f}' if isinstance(n_buy, (int, float, np.floating)) else '尚未形成明確頸線'}
- **波段加碼參考價**：{add_price:.1f}，條件：帶量突破前高且回測不破
- **波段第一目標價**：{f'{swing_target_1:.1f}' if isinstance(swing_target_1, (int, float, np.floating)) else '尚未形成'}
- **波段強勢目標價**：{swing_target_2:.1f}
- **波段停損價**：{min(curr_p * 0.94, swing_stop if swing_stop is not None else curr_p):.1f}

---

#### 📉 分批進場策略建議(僅適用於正在漲的時候)
1. **第一批進場 (30%)**
   - **買入參考價：{curr_p:.1f}**
   - 條件：股價站穩目前價位，且沒有跌破短線停損。
   - 對應停損：**{dynamic_stop['price']:.1f}**

2. **第二批加碼 (40%)**
   - **加碼參考價：{recent_high:.1f}**
   - 條件：帶量突破近期高點，且回測不破。
   - 回測不破可視為突破確認，不建議未突破前追高加碼。

3. **最後倉位 (30%)**
   - **補齊參考價：{t1:.1f}**
   - 條件：AI 噴發機率 > 60%，且市場結構維持 **HH+HL**。
   - 若突破第一目標後量能續強，才考慮補齊最後倉位。

---

#### 🌊 波浪與估值
- **市場本質**：{wave_info}
- **EPS**：{f"{eps:.2f}" if eps else "數據不足"}
- **20x 本益比估值目標**：{f"${fin_target:.1f}" if fin_target else "數據不足"}
""")
