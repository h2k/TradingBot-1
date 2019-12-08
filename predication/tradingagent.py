import copy
import pickle
import numpy as np
from predication import (
    softmax,
    get_state
)
from stocks import StockDatabase
from predication import TraidingAiStrategy


class TradingAgent(object):
    """
    Represents a trading agent
    Trading agent apply the actions advised by the model for the user
    """
    def __init__(self, model, timeseries=None, skip=1, initial_money=1000, real_trend=None, minmax=None,
                 symbol=None):
        self.db = StockDatabase("84WDI082Z0HOREL6")
        # User data
        self._capital = 0
        self.user = None
        self._scaled_capital = None
        self._queue = []
        self._inventory = []
        # DON'T TOUCH (sklearn properties)
        self.population_size = 20
        self.sigma = 0.1
        self.learning_rate = 0.03
        # Leave 20 days to analyze trend before taking actions or else not enough data
        self.window_size = 20
        self.model = model
        self.timeseries = timeseries
        # Always skip one day
        self.skip = skip
        # Used by test method
        self.real_trend = real_trend
        self.initial_money = initial_money
        self.es = TraidingAiStrategy(self.model.weights, self.get_reward, self.population_size, self.sigma,
                                     self.learning_rate)
        self.minmax = minmax
        self.symbol = symbol
        self._initiate()

    def load_window(self, data):
        """
        Load last self.window_size days of timeseries to fill window size
        """
        # Leave last day for analysis
        data = data.iloc[:-1].tail(self.window_size)
        close = data['4. close'].tolist()
        volume = data['5. volume'].tolist()
        for i in range(self.window_size):
            day_data = [close[i], volume[i]]
            scaled_data = self.minmax.transform([day_data])[0]
            self._queue.append(scaled_data)

    def load_user_info(self, user):
        """
        Load data of the user to trade according to his inventory / capital
        """
        self.user = user
        self._capital = user.invested_balance
        self._scaled_capital = self.minmax.transform([[self._capital, 2]])[0, 0]
        if len(user.inventory):
            # Load user inventory price for traded symbol
            items = [item["stock"] for item in user.inventory if item["stock"]["symbol"] == self.symbol]
            if len(items):
                for stock in self.db.stocks:
                    if stock.symbol == self.symbol:
                        self._inventory.extend([stock.price for i in range(round(items[0]["quantity"]))])
        else:
            self._inventory = []

    def _initiate(self):
        """
        Initiate trading agent model
        :return:
        """
        if self.timeseries is not None:
            self.trend = self.timeseries[0]
            self._mean = np.mean(self.trend)
            self._std = np.std(self.trend)
        self._inventory = []
        self._capital = self.initial_money
        self._queue = []
        self._scaled_capital = self.minmax.transform([[self._capital, 2]])[0, 0]

    def reset_capital(self, capital):
        if capital:
            self._capital = capital
        self._scaled_capital = self.minmax.transform([[self._capital, 2]])[0, 0]
        self._queue = []
        self._inventory = []

    def save_model(self, filename=None):
        """
        Save training to be reused
        :param filename Where to save, default to trade_model.pkl
        :return:
        """
        copy_model = copy.deepcopy(self.model)
        filename = filename if filename is not None else 'trade_model.pkl'
        with open(filename, 'wb+') as file:
            pickle.dump(copy_model, file)

    def trade(self, data):
        """
        Initiate a trade for a user

        :param data Must be pd of format [Close, Volume]
        """
        scaled= self.minmax.transform([data])[0]
        real_close = data[0]
        close = scaled[0]
        if len(self._queue) >= self.window_size:
            self._queue.pop(0)
        self._queue.append(scaled)
        # Model is still training, return nothing
        if len(self._queue) < self.window_size:
            return None
        state = self.retrive_state(self.window_size - 1, self._inventory, self._scaled_capital,
                                   timeseries=np.array(self._queue).T.tolist())
        action, prob = self.act_softmax(state)
        # BUY
        if action == 1 and self._scaled_capital >= close:
            self._inventory.append(close)
            self._scaled_capital -= close
            self._capital -= real_close
            # Update user invested balance
            if self.user is not None:
                # Add this action to his inventory
                self.user.update_inventory(self.symbol, quantity=1, income=real_close, action="buy")
            return {
                'symbol': self.symbol,
                'action': 'buy',
                'quantity': 1,
                'income': real_close
            }
        # SELL
        elif action == 2 and len(self._inventory):
            self._inventory.pop(0)
            self._scaled_capital += close
            self._capital += real_close
            if self.user is not None:
                # Remove this action from his inventory
                self.user.update_inventory(self.symbol, quantity=1, income=real_close, action="sell")
            return {
                'symbol': self.symbol,
                'action': 'sell',
                'amount': 1,
                'income': real_close
            }
        else:
            # Nothing
            return None

    def change_data(self, timeseries, skip, initial_money, real_trend, minmax):
        """
        Reload data
        """
        self.timeseries = timeseries
        self.skip = skip
        self.initial_money = initial_money
        self.real_trend = real_trend
        self.minmax = minmax
        self._initiate()

    def act(self, sequence):
        """
        Predict next move
        """
        decision = self.model.predict(np.array(sequence))
        return np.argmax(decision[0])

    def act_softmax(self, sequence):
        """
        Use softmax to predict next action
        """
        decision = self.model.predict(np.array(sequence))
        return np.argmax(decision[0]), softmax(decision)[0]

    def retrive_state(self, t, inventory, capital, timeseries):
        """
        Get decision tree state
        """
        state = get_state(timeseries, t)
        size_inventory = len(inventory)
        if size_inventory:
            mean_inventory = np.mean(inventory)
        else:
            mean_inventory = 0
        z_inventory = (mean_inventory - self._mean) / self._std
        z_capital = (capital - self._mean) / self._std
        # Concat parameters in a matrix on one axis
        concat_parameters = np.concatenate([state, [[size_inventory, z_inventory, z_capital]]], axis=1)
        return concat_parameters

    def get_reward(self, weights):
        """
        Calculate rewards based on weights
        The higher the rewards, the higher the profit made by the AI
        """
        initial_money = self._scaled_capital
        starting_money = initial_money
        investments = []
        user_inventory = []
        self.model.weights = weights
        state = self.retrive_state(0, user_inventory, starting_money, self.timeseries)
        for t in range(0, len(self.trend) - 1, self.skip):
            decision = self.act(state)
            # Buy
            if decision == 1 and starting_money >= self.trend[t]:
                starting_money -= self.trend[t]
                user_inventory.append(self.trend[t])
            elif decision == 2 and len(user_inventory):
                starting_money += self.trend[t]
                bought_price = user_inventory.pop(0)
                invest = ((self.trend[t] - bought_price) / bought_price) * 100
                investments.append(invest)
            state = self.retrive_state(t + 1, user_inventory, starting_money, self.timeseries)
        invests = np.mean(investments)
        if np.isnan(invests):
            invests = 0
        # Give us a percentage
        score = (starting_money - initial_money) / initial_money * 100
        # Investments made are better than a better score (we want more investments rather than a high score)
        # The more investments, the more the AI is responsive to change
        # See Alex for validation / tweek
        return invests * 0.7 + score * 0.3

    def fit(self, iterations):
        """
        Train model for nb iterations
        """
        self.es.train(iterations)

    def test_trade(self):
        """
        Used by test method to run the whole timeseries through the model

        Return the global stats of the current investing session containing buy, sell, profit, capital
        """
        initial_money = self._scaled_capital
        starting_money = initial_money
        real_initial_money = self.initial_money
        real_starting_money = self.initial_money
        # Keep two inventories because could be useful outside test or multiple portfolio
        inventory = []
        real_inventory = []
        state = self.retrive_state(0, inventory, starting_money, self.timeseries)
        sells = []
        buys = []
        for t in range(0, len(self.trend) - 1, self.skip):
            action, prob = self.act_softmax(state)
            if action == 1 and starting_money >= self.trend[t] and t < (len(self.trend) - 1 - self.window_size):
                inventory.append(self.trend[t])
                real_inventory.append(self.real_trend[t])
                real_starting_money -= self.real_trend[t]
                starting_money -= self.trend[t]
                buys.append(t)
            elif action == 2 and len(inventory):
                inventory.pop(0)
                real_inventory.pop(0)
                starting_money += self.trend[t]
                real_starting_money += self.real_trend[t]
                sells.append(t)
            state = self.retrive_state(t + 1, inventory, starting_money, self.timeseries)
        invest = ((real_starting_money - real_initial_money) / real_initial_money) * 100
        total_gains = real_starting_money - real_initial_money
        return buys, sells, total_gains, invest