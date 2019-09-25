import os
import csv
import time
import urllib
from pathlib import Path
import datetime
from alpha_vantage import TimeSeries, CryptoCurrencies
from .stock import Stock


class StockDatabase(object):
    """Download all the available values of known stock"""
    nasdaq_url = "https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download"
    stock_filename = "stock_estimations/companylist"

    def __init__(self, api_key):
        self.ts = TimeSeries(key=api_key, output_format='pandas', indexing_type='date')
        self.cc = CryptoCurrencies(key=api_key, output_format='pandas', indexing_type='date')
        self.nasdaq_stockfile = f'{self.stock_filename}-{datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d")}.csv'
        self.stocks = []
        os.makedirs("stock_estimations", exist_ok=True)
        self._download_nasdaq_stocks()
        with open(self.nasdaq_stockfile, "r") as file:
            csv_content = csv.reader(file, delimiter=',', quotechar='"')
            next(csv_content, None)
            for row in csv_content:
                row = list(filter(None, row))
                self.stocks.append(Stock.from_nasdaq(row))

    def generate_stock_history(self, stock, time_series="day", output_size="full", interval=None, is_crypto=False):
        if time_series == "intraday":
            data, _ = self.ts.get_intraday(symbol=stock.symbol, outputsize=output_size, interval=interval) if not is_crypto else self.cc.get_digital_currency_intraday(symbol=stock.symbol, output_size=output_size)
        if time_series == "day":
            data, _ = self.ts.get_daily(symbol=stock.symbol, outputsize=output_size) if not is_crypto else self.cc.get_digital_currency_daily(symbol=stock.symbol, market="USD")
        if time_series == "week":
            data, _ = self.ts.get_weekly(symbol=stock.symbol) if not is_crypto else self.cc.get_digital_currency_weekly(symbol=stock.symbol, market="USD")
        if time_series == "month":
            data, _ = self.ts.get_monthly(symbol=stock.symbol) if not is_crypto else self.cc.get_digital_currency_monthly(symbol=stock.symbol, market="USD")
        directory = f'stock_estimations/{stock.symbol}/{datetime.datetime.today().strftime("%d-%m-%Y")}'
        filename = f'{directory}/{stock.symbol}-prices.csv'
        Path(directory).mkdir(parents=True, exist_ok=True)
        with open(filename, 'w+') as out:
            for index, row in data.iterrows():
                out.write(f'{index.replace("-", "")},{row.iloc[0]}\n')
        return filename

    def _download_nasdaq_stocks(self):
        yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
        yesterday_file = f'{self.stock_filename}-{yesterday.strftime("%Y%m%d")}.csv'
        if os.path.exists(yesterday_file):
            os.remove(yesterday_file)
        if os.path.exists(self.nasdaq_stockfile):
            return
        else:
            urllib.request.urlretrieve(self.nasdaq_url, self.nasdaq_stockfile)
