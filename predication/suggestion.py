import os
import base64
import tempfile
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce
from alpha_vantage import TimeSeries
from sklearn.linear_model import LinearRegression


class SuggestionEngine(object):
    """
    Basic suggestion using stock volatility and mean returns
    """
    def __init__(self, symbols):
        self.ts = TimeSeries(key="84WDI082Z0HOREL6", output_format='pandas', indexing_type='date')
        sns.set()
        self.MEAN_ERROR = 0.1
        self.symbols = symbols
        self.stocks_data = None
        self.retrieve_stock_values()
        self.low_risk_range = (0, 10)
        self.medium_risk_range = (10, 20)
        self.high_risk_range = (20, 999)

    def suggest(self):
        """
        Suggest a stock
        :return:
        """
        if self.stocks_data.empty:
            self.retrieve_stock_values()
        returns = self.stocks_data.pct_change()
        mean_daily_returns = returns.mean()
        #
        volatilities = returns.std()
        # https://www.fool.com/knowledge-center/how-to-calculate-annualized-volatility.aspx
        combine = pd.DataFrame({'retour': mean_daily_returns * 252,
                                'volatilite du stock': volatilities * 252})
        ret = {
            "buy": {
                "low": [],
                "medium": [],
                "high": []
            },
            "sell": [],
            "result": None
        }
        x = combine["volatilite du stock"].values[:, np.newaxis]
        y = combine["retour"].values
        model = LinearRegression()
        model.fit(x, y)
        linear_reg = model.predict(x)
        # y = ax + b
        combine["Distance"] = y - linear_reg
        # https://seaborn.pydata.org/generated/seaborn.jointplot.html
        g = sns.jointplot("volatilite du stock", "retour", data=combine, kind="reg", height=7)
        for i in range(combine.shape[0]):
            # Under linear regression
            if combine.iloc[i, 2] - self.MEAN_ERROR > 0:
                ret["sell"].append(self.symbols[i])
            else:
                # Check risk threshold
                risk = combine.iloc[i, 1]
                if self.low_risk_range[0] <= risk <= self.low_risk_range[1]:
                    ret["buy"]["low"].append(self.symbols[i])
                elif self.medium_risk_range[0] <= risk <= self.medium_risk_range[1]:
                    ret["buy"]["medium"].append(self.symbols[i])
                else:
                    ret["buy"]["high"].append(self.symbols[i])
            plt.annotate(self.symbols[i], (combine.iloc[i, 1], combine.iloc[i, 0]))
        filename = f'{next(tempfile._get_candidate_names())}-plot.png'
        plt.savefig(filename)
        with open(filename, 'rb') as image_file:
            ret["result"] = base64.b64encode(image_file.read()).decode('utf-8')
        print(ret)
        os.remove(filename)
        return ret

    def retrieve_stock_values(self):
        """
        Retrieve stocks values using AlphaVantage wrapper
        Limit to MAX_STOCKS stocks name at a time
        :return:
        """
        filenames = []
        for symbol in self.symbols:
            data, _ = self.ts.get_daily(symbol=symbol, outputsize="full")
            filename = f'{next(tempfile._get_candidate_names())}.csv'
            filenames.append(filename)
            data.to_csv(path_or_buf=filename)
        dfs = [pd.read_csv(file)[['date', '4. close']] for file in filenames]
        self.stocks_data = reduce(lambda left, right: pd.merge(left, right, on='date'), dfs).iloc[:, 1:]
        [os.remove(f) for f in filenames]
        return True
