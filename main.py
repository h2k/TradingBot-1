from datetime import datetime
import json
from predication import Engine
from stocks import StockDatabase

API_KEY_TEST = '84WDI082Z0HOREL6'
INTERVAL = "60min"
TIME_SERIES = "day"
TEST_STOCKS = ["AMZN"]


def main():
    db = StockDatabase(API_KEY_TEST)
    for stock in db.stocks:
        if stock.symbol in TEST_STOCKS:
            filename = db.generate_stock_history(stock, time_series=TIME_SERIES, output_size="full", interval=INTERVAL)
            print(f'[{datetime.now()}][+] Generated data for {stock.name} ({stock.symbol}) - {stock.sector}')
            engine = Engine(stock, filename)
            print(f'[{datetime.now()}][+] Starting training model...')
            engine.train_model(save_results=True)


if __name__ == "__main__":
    main()