import os
import base64
import pickle
import time
import numpy as np
from predication import *
from alpha_vantage import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from predication import (
    TradingAgent,
    SuggestionEngine
)
from users import User


def predict_error(job, exc_type, exc_value, traceback):
    print(job, exc_type, exc_value, traceback)


def predict(stock, filename):
    engine = Engine(stock, filename)
    filepath = engine.train_model(save_results=True)
    with open(filepath, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def suggest(symbols):
    suggest_engine = SuggestionEngine(symbols)
    return suggest_engine.suggest()


def trade(bearer):
    """
    Trade for a user
    """
    ts = TimeSeries("84WDI082Z0HOREL6", output_format="pandas")
    # Load user
    user = User(bearer)
    suggest_engine = SuggestionEngine(symbols=["MSFT", "AAPL", "AMD", "CSCO"])
    suggestion = suggest_engine.suggest()
    # Load model
    model = None
    save_file = os.path.join(os.getcwd(), "trade_model.pkl")
    ret = {
        "suggested": {
            "buy": [],
            "sell": [],
        },
        "checked_symbols": [],
        "actions": []
    }
    with open(save_file, "rb") as model_file:
        model = pickle.load(model_file)
    # Append suggested buy trade to list of checks
    print(f"User has risk {user.get_risk()}")
    symbols_to_check = []
    if len(suggestion["buy"][user.get_risk()]):
        symbols_to_check.extend(suggestion["buy"][user.get_risk()])
        ret["suggested"]["buy"] = suggestion["buy"][user.get_risk()]
    if len(suggestion["sell"]):
        symbols_to_check.extend(suggestion["sell"])
        ret["suggested"]["sell"] = suggestion["sell"]
    if len(user.inventory):
        symbols_to_check.extend([item["stock"]["symbol"] for item in user.inventory])
    # Run a check on all the user inventory + suggested trade offers (both sell and buy)
    ret["checked_symbols"] = symbols_to_check
    time.sleep(60)
    for symbol in symbols_to_check:
        print(f"TradingAgent: Checking {symbol}...")
        data, _ = ts.get_daily(symbol=symbol, outputsize="compact")
        parameters = [data['4. close'].tolist(), data['5. volume'].tolist()]
        minmax = MinMaxScaler(feature_range=(100, 200)).fit(np.array(parameters).T)
        trend = data['4. close'].tolist()
        scaled_parameters = minmax.transform(np.array(parameters).T).T.tolist()
        agent = TradingAgent(model, timeseries=scaled_parameters, real_trend=trend, minmax=minmax, symbol=symbol)
        # Load user inventory
        agent.load_user_info(user)
        # Feed 20 days
        agent.load_window(data.iloc[0:-20])
        last_data = data.tail(1)
        print(f"Data loaded for {symbol}, taking decision...")
        action = agent.trade([last_data["4. close"].tolist()[0], last_data["5. volume"].tolist()[0]])
        if action is not None:
            ret["actions"].append(action)
        print(f"Action for {symbol} : {action}")
    return ret