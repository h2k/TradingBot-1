import os
import unittest
from predication import Engine
from stocks import StockDatabase

API_KEY = "84WDI082Z0HOREL6"


class TestTrading(unittest.TestCase):
    def testDownloadNasdaq(self):
        db = StockDatabase(API_KEY)
        self.assertTrue(os.path.exists(db.nasdaq_stockfile), "The NASDAQ file couldn't be downloaded")

    def testStockAnalysis(self):
        db = StockDatabase(API_KEY)
        stock = next((stock for stock in db.stocks if stock.symbol == "MSFT"), None)
        self.assertIsNotNone(stock, "No stock was found")
        filename = db.generate_stock_history(stock, time_series="day", output_size="full", interval="60min")
        directory = os.path.dirname(filename)
        engine = Engine(stock, filename)
        engine.train_model(save_results=True)
        self.assertTrue(os.path.exists(f'{directory}/{stock.symbol}-plot.png'), "No graph was created")


if __name__ == '__main__':
    unittest.main()
