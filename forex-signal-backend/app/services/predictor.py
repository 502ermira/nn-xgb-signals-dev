import pandas_ta as ta

def make_prediction(df):
    df = df.copy()

    # Indicators
    df["rsi"] = ta.rsi(df["close"], length=14)
    df = df.join(ta.macd(df["close"]))
    df = df.join(ta.bbands(df["close"], length=20))
    df = df.join(ta.stoch(df["high"], df["low"], df["close"]))
    df["ema20"] = ta.ema(df["close"], length=20)
    df["ema50"] = ta.ema(df["close"], length=50)
    df["adx"] = ta.adx(df["high"], df["low"], df["close"])["ADX_14"]
    df["cci"] = ta.cci(df["high"], df["low"], df["close"], length=20)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    latest = df.iloc[-1]
    signal_reasons = []
    signal = "HOLD"
    # RSI logic
    if latest["rsi"] < 30:
        signal_reasons.append("RSI indicates oversold")
    elif latest["rsi"] > 70:
        signal_reasons.append("RSI indicates overbought")

    # MACD logic
    if latest["MACD_12_26_9"] > latest["MACDs_12_26_9"]:
        signal_reasons.append("MACD bullish crossover")
    elif latest["MACD_12_26_9"] < latest["MACDs_12_26_9"]:
        signal_reasons.append("MACD bearish crossover")

    # Bollinger Band logic
    if latest["close"] < latest["BBL_20_2.0"]:
        signal_reasons.append("Price below lower Bollinger Band (oversold)")
    elif latest["close"] > latest["BBU_20_2.0"]:
        signal_reasons.append("Price above upper Bollinger Band (overbought)")

    # Stochastic logic
    if latest["STOCHk_14_3_3"] < 20:
        signal_reasons.append("Stochastic indicates oversold")
    elif latest["STOCHk_14_3_3"] > 80:
        signal_reasons.append("Stochastic indicates overbought")

    # EMA crossover logic
    if latest["ema20"] > latest["ema50"]:
        signal_reasons.append("EMA20 above EMA50 (bullish trend)")
    elif latest["ema20"] < latest["ema50"]:
        signal_reasons.append("EMA20 below EMA50 (bearish trend)")

    # ADX trend strength
    if latest["adx"] > 25:
        signal_reasons.append("Strong trend detected (ADX > 25)")

    # CCI logic
    if latest["cci"] > 100:
        signal_reasons.append("CCI indicates strong buying pressure")
    elif latest["cci"] < -100:
        signal_reasons.append("CCI indicates strong selling pressure")

    buy_signals = [
        "RSI indicates oversold",
        "MACD bullish crossover",
        "Price below lower Bollinger Band (oversold)",
        "Stochastic indicates oversold",
        "EMA20 above EMA50 (bullish trend)",
        "CCI indicates strong buying pressure"
    ]

    sell_signals = [
        "RSI indicates overbought",
        "MACD bearish crossover",
        "Price above upper Bollinger Band (overbought)",
        "Stochastic indicates overbought",
        "EMA20 below EMA50 (bearish trend)",
        "CCI indicates strong selling pressure"
    ]

    if any(r in signal_reasons for r in buy_signals) and not any(r in signal_reasons for r in sell_signals):
        signal = "BUY"
    elif any(r in signal_reasons for r in sell_signals) and not any(r in signal_reasons for r in buy_signals):
        signal = "SELL"

    return {
        "signal": signal,
        "rsi": round(latest["rsi"], 2),
        "macd": round(latest["MACD_12_26_9"], 4),
        "macd_signal": round(latest["MACDs_12_26_9"], 4),
        "close": round(latest["close"], 4),
        "bollinger_upper": round(latest["BBU_20_2.0"], 4),
        "bollinger_lower": round(latest["BBL_20_2.0"], 4),
        "stochastic_k": round(latest["STOCHk_14_3_3"], 2),
        "stochastic_d": round(latest["STOCHd_14_3_3"], 2),
        "ema20": round(latest["ema20"], 4),
        "ema50": round(latest["ema50"], 4),
        "adx": round(latest["adx"], 2),
        "cci": round(latest["cci"], 2),
        "atr": round(latest["atr"], 4),
        "reason": signal_reasons,
    }