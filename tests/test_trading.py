import os
import string
import random
import unittest
import pickle
import numpy as np
import matplotlib.pyplot as plt
from stocks import StockDatabase
from alpha_vantage import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from predication import (
    Model,
    TradingAgent,
    SuggestionEngine
)
from users import User

API_KEY = "84WDI082Z0HOREL6"
DEST_PLOT_DIR = f'{os.getcwd()}/results'


class TestTrading(unittest.TestCase):
    """
    Test class for all trading issues
    """
    def setUp(self) -> None:
        os.makedirs(DEST_PLOT_DIR, exist_ok=True)

    def testDownloadNasdaq(self):
        db = StockDatabase(API_KEY)
        self.assertTrue(os.path.exists(db.nasdaq_stockfile), "The NASDAQ file couldn't be downloaded")

    def testTradingReloadSavedSimulateSmallInventory(self):
        """
        Test trading by loading data from a user for multiple days

        User have a small inventory
        """
        symbol = "AAPL"
        user = User(None)
        start_capital = 1500
        model = None
        # Create a fake user
        email = f"{''.join(random.choices(string.ascii_uppercase + string.digits, k=10))}@test.com"
        password = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        user.create_user("Test", "Test", email, password)
        user.login(email, password)
        self.assertIsNotNone(user.bearer, "Couldn't connect as fake user on API")
        print(f"[*] Testing for user '{email}' with password '{password}'")
        # User has 1500 invested and 1500 available
        self.assertTrue(user.update_balance(start_capital))
        self.assertTrue(user.update_invested_balance(start_capital))
        self.assertEqual(0, user.balance, "User balance wasn't updated properly")
        self.assertEqual(start_capital, user.invested_balance, "User invested balance wasn't updated properly")
        # Load model
        save_file = os.path.join(os.getcwd(), "trade_model.pkl")
        with open(save_file, "rb") as model_file:
            model = pickle.load(model_file)
        self.assertIsNotNone(model, "Couldn't load sklearn model")
        print("[+] Model loaded !")
        ts = TimeSeries(API_KEY, output_format="pandas")
        data, _ = ts.get_daily(symbol=symbol, outputsize="full")
        self.assertIsNotNone(data, "Couldn't retrieve data from AlphaVantage")
        parameters = [data['4. close'].tolist(), data['5. volume'].tolist()]
        minmax = MinMaxScaler(feature_range=(100, 200)).fit(np.array(parameters).T)
        trend = data['4. close'].tolist()
        scaled_parameters = minmax.transform(np.array(parameters).T).T.tolist()
        agent = TradingAgent(model, timeseries=scaled_parameters, real_trend=trend, minmax=minmax, symbol=symbol)
        # User has a small inventory on said stock
        stock_price = data.iloc[0:-120].tail(1)["4. close"].tolist()[0]
        print(f"Stock {symbol} worth {stock_price}/ea")
        user.update_balance(start_capital + (stock_price * 2))
        user.update_invested_balance(start_capital + (stock_price * 2))
        user.update_inventory(symbol, 2, stock_price, "buy")
        # Load user inventory and balance -120
        agent.load_user_info(user)
        print("[+] User info loaded !")
        # Feed 20 days from 100 days ago (so D-120 to D-100)
        agent.load_window(data.iloc[0:-120])
        # Run actions from 100 days to today
        data = data.tail(100)
        buys = []
        sells = []
        for i in range(100):
            print(f"Day n°{i} : Invested : {user.invested_balance} | Inventory : {user.inventory}")
            last_data = data.iloc[i]
            self.assertIsNotNone(last_data)
            action = agent.trade([last_data["4. close"].tolist(), last_data["5. volume"].tolist()])
            if action is not None:
                if action['action'] == 'buy':
                    buys.append(i)
                else:
                    sells.append(i)
        # Create figure to have multiple plots
        fig = plt.figure(figsize=(15, 5))
        plt.plot(data['4. close'], color='r', lw=2.0, label=f'Valeur du stock {symbol}')
        plt.plot(data['4. close'], '^', markersize=8, color='m', label="Signal d'achat", markevery=buys)
        plt.plot(data['4. close'], 'v', markersize=8, color='k', label='Signal de vente', markevery=sells)
        print(f"Started at {start_capital}, now sitting at total {user.balance + user.invested_balance}")
        invest = (((user.balance + user.invested_balance) - start_capital) / start_capital) * 100
        plt.title(f'{symbol} - Simulation pour user sur 100 jours (Gain total : {round(invest)}%)')
        plt.legend()
        plt.savefig(f'{DEST_PLOT_DIR}/{symbol}-user-simulated-trade')
        plt.close(fig)

    def testTradingReloadSavedSimulateNoInventory(self):
        """
        Test trading by loading data from a user for multiple days

        User DOESN'T HAVE ANY INVENTORY ITEMS
        """
        symbol = "AAPL"
        user = User(None)
        start_capital = 1500
        model = None
        # Create a fake user
        email = f"{''.join(random.choices(string.ascii_uppercase + string.digits, k=10))}@test.com"
        password = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        user.create_user("Test", "Test", email, password)
        user.login(email, password)
        self.assertIsNotNone(user.bearer, "Couldn't connect as fake user on API")
        print(f"[*] Testing for user '{email}' with password '{password}'")
        # User has 1500 invested and 1500 available
        self.assertTrue(user.update_balance(start_capital))
        self.assertTrue(user.update_invested_balance(start_capital))
        self.assertEqual(0, user.balance, "User balance wasn't updated properly")
        self.assertEqual(start_capital, user.invested_balance, "User invested balance wasn't updated properly")
        # Load model
        save_file = os.path.join(os.getcwd(), "trade_model.pkl")
        with open(save_file, "rb") as model_file:
            model = pickle.load(model_file)
        self.assertIsNotNone(model, "Couldn't load sklearn model")
        print("[+] Model loaded !")
        ts = TimeSeries(API_KEY, output_format="pandas")
        data, _ = ts.get_daily(symbol=symbol, outputsize="full")
        self.assertIsNotNone(data, "Couldn't retrieve data from AlphaVantage")
        parameters = [data['4. close'].tolist(), data['5. volume'].tolist()]
        minmax = MinMaxScaler(feature_range=(100, 200)).fit(np.array(parameters).T)
        trend = data['4. close'].tolist()
        scaled_parameters = minmax.transform(np.array(parameters).T).T.tolist()
        agent = TradingAgent(model, timeseries=scaled_parameters, real_trend=trend, minmax=minmax, symbol=symbol)
        # Load user inventory and balance -120
        agent.load_user_info(user)
        print("[+] User info loaded !")
        # Feed 20 days from 100 days ago (so D-120 to D-100)
        agent.load_window(data.iloc[0:-120])
        # Run actions from 100 days to today
        data = data.tail(100)
        buys = []
        sells = []
        for i in range(100):
            print(f"Day n°{i} : Invested : {user.invested_balance} | Inventory : {user.inventory}")
            last_data = data.iloc[i]
            self.assertIsNotNone(last_data)
            action = agent.trade([last_data["4. close"].tolist(), last_data["5. volume"].tolist()])
            if action is not None:
                if action['action'] == 'buy':
                    buys.append(i)
                else:
                    sells.append(i)
        # Create figure to have multiple plots
        fig = plt.figure(figsize=(15, 5))
        plt.plot(data['4. close'], color='r', lw=2.0, label=f'Valeur du stock {symbol}')
        plt.plot(data['4. close'], '^', markersize=8, color='m', label="Signal d'achat", markevery=buys)
        plt.plot(data['4. close'], 'v', markersize=8, color='k', label='Signal de vente', markevery=sells)
        print(f"Started at {start_capital}, now sitting at total {user.balance + user.invested_balance}")
        invest = (((user.balance + user.invested_balance) - start_capital) / start_capital) * 100
        plt.title(f'{symbol} - Simulation pour user sur 100 jours (Gain total : {round(invest)}%)')
        plt.legend()
        plt.savefig(f'{DEST_PLOT_DIR}/{symbol}-user-simulated-trade')
        plt.close(fig)

    def testSuggestion(self):
        """
        Verify that suggestion engine is working properly
        """
        symbols = ["AMD", "GOOG", "ROKU", "MSFT", "SBUX"]
        suggest_engine = SuggestionEngine(symbols)
        ret = suggest_engine.suggest()
        self.assertIsNotNone(ret, "AI suggested something")
        self.assertTrue("buy" in ret, "Wrong suggestion format")

    def testTradingManually(self):
        """
        Test trading stocks analysis and manually verify results
        """
        ts = TimeSeries(API_KEY, output_format="pandas")
        # Containing old historical data (picked from https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
        training_stocks = ["AMD", "FSV", "ROKU", "TMUS", "FTNT", "KHC", "TRIP", "TXN", "LYFT", "FSV", "SINA", "BIIB"]
        # Interesting stocks to tests onto
        up_trends = ["MSFT", "AAPL", "FB", "ADBE", "CTXS"]
        down_trends = ["ALGN", "MYL", "AAL"]
        test_stocks = up_trends + down_trends
        skip = 1
        layer_size = 500
        # 1500 dollars should always be enough
        initial_money = 1500
        output_size = 3
        model = Model(79, layer_size, output_size)
        model_initialized = False
        agent = None
        # We need to train our model first
        for stock in training_stocks:
            print(f"Training on {stock} with {initial_money}$")
            data, _ = ts.get_daily(symbol=stock, outputsize="full")
            self.assertIsNotNone(data, "Couldn't retrieve data from Alphavantage")
            # Take a full year of trading
            data = data.tail(253)
            print(data)
            trend = data['4. close'].tolist()
            parameters = [data['4. close'].tolist(), data['5. volume'].tolist()]
            minmax = MinMaxScaler(feature_range=(100, 200)).fit(np.array(parameters).T)
            scaled_parameters = minmax.transform(np.array(parameters).T).T.tolist()
            if not model_initialized:
                agent = TradingAgent(model, scaled_parameters, skip, initial_money, trend, minmax)
                # Enable debug mode for test purpose
                agent.es.debug = True
                model_initialized = True
            else:
                # Feed new training data to model without recreating one
                agent.change_data(scaled_parameters, skip, initial_money, trend, minmax)
            # Use 100 epoch for best result
            agent.fit(iterations=100)
        # Model is trained, now test on non-trained stocks
        for stock in test_stocks:
            data, _ = ts.get_daily(symbol=stock, outputsize="full")
            self.assertIsNotNone(data, "Couldn't retrieve data from Alphavantage")
            # Take a full year of trading
            data = data.tail(253)
            data['5. volume'] = data['5. volume'].astype(int)
            trend = data['4. close'].tolist()
            parameters = [data['4. close'].tolist(), data['5. volume'].tolist()]
            minmax = MinMaxScaler(feature_range=(100, 200)).fit(np.array(parameters).T)
            scaled_parameters = minmax.transform(np.array(parameters).T).T.tolist()
            agent.change_data(scaled_parameters, skip, initial_money, trend, minmax)
            buys, sells, total_gain, invest = agent.test_trade()
            # Create figure to have multiple plots
            fig = plt.figure(figsize=(15, 5))
            plt.plot(data['4. close'], color='r', lw=2.0, label=f'Valeur du stock {stock}')
            plt.plot(data['4. close'], '^', markersize=8, color='m', label="Signal d'achat", markevery=buys)
            plt.plot(data['4. close'], 'v', markersize=8, color='k', label='Signal de vente', markevery=sells)
            plt.title(f'{stock} - Gain total : {invest}%')
            plt.legend()
            plt.savefig(f'{DEST_PLOT_DIR}/{stock}')
            plt.close(fig)
        agent.save_model()


if __name__ == '__main__':
    unittest.main()
