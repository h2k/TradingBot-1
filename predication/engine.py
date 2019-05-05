import os
import json
import numpy as np
import pandas as pd
from math import sqrt
from datetime import datetime
from sentiment import TwitterSentiment
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler


class Engine(object):
    def __init__(self, stock, price_filename: str):
        self.stock = stock
        self.dir = os.path.dirname(price_filename)
        self.price_filename = price_filename
        self.sentiment_filename = price_filename.replace("prices", "sentiment")
        self.merged_filename = price_filename.replace("prices", "merged")
        with open('config.json', 'r') as json_file:
            config = json.load(json_file)
            self.twitter = TwitterSentiment(config["twitter"])

    def generate_stock_sentiment(self, stock):
        if not os.path.exists(self.sentiment_filename):
            tweets = self.twitter.get_tweets(f'{stock.name}, {stock.symbol}', since=stock.entry_year, count=1000)
            with open(self.sentiment_filename, 'w') as tweet_file:
                for tweet in tweets:
                    tweet_file.write(tweet.csv_dump())

    def _merge_files(self):
        file = pd.read_csv(self.price_filename)
        file.columns = ["Time", "Price"]
        sent = pd.read_csv(self.sentiment_filename)
        sent.columns = ["Time", "Sentiment"]
        merged = sent.merge(file, left_index=False, right_index=False, how="inner")
        merged = merged.replace(to_replace='None', value=np.nan).dropna()
        merged = merged.sort_values(by=["Time"])
        merged.to_csv(self.merged_filename)

    @staticmethod
    def create_dataset(dataset, look_back, sentiment, sent=False):
        data_x, data_y = [], []
        for i in range(len(dataset) - look_back):
            if i >= look_back:
                a = dataset[i - look_back:i + 1, 0]
                a = a.tolist()
                if sent:
                    a.append(sentiment[i].tolist()[0])
                data_x.append(a)
                data_y.append(dataset[i + look_back, 0])
        return np.array(data_x), np.array(data_y)

    def train_model(self, extra_indicators=None, save_results=False):
        self.generate_stock_sentiment(self.stock)
        self._merge_files()
        data = pd.read_csv(self.merged_filename)
        datag = data[['Price', 'Sentiment']].groupby(data['Time']).mean()
        values = datag['Price'].values.reshape(-1, 1)
        sentiment = datag['Sentiment'].values.reshape(-1, 1)
        values = values.astype('float32')
        sentiment = sentiment.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        train_size = int(len(scaled) * 0.7)
        train, test = scaled[0:train_size, :], scaled[train_size:len(scaled), :]
        look_back = 2
        train_x, train_y = self.create_dataset(train, look_back, sentiment[0:train_size], sent=True)
        test_x, test_y = self.create_dataset(test, look_back, sentiment[train_size:len(scaled)], sent=True)
        train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
        test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
        model = Sequential()
        model.add(LSTM(100, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
        model.add(LSTM(100))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam', metrics=["accuracy"])
        history = model.fit(train_x, train_y, epochs=500, batch_size=100, validation_data=(test_x, test_y), verbose=0, shuffle=False)
        yhat = model.predict(test_x)
        yhat_inverse_sent = scaler.inverse_transform(yhat.reshape(-1, 1))
        test_y_inverse_sent = scaler.inverse_transform(test_y.reshape(-1, 1))
        rmse_sent = sqrt(mean_squared_error(test_y_inverse_sent, yhat_inverse_sent))
        yhat = model.predict(test_x)
        test_y = np.reshape(test_y, (-1, 1))
        test_y = scaler.inverse_transform(test_y)
        yhat = scaler.inverse_transform(yhat)
        print(f'[{datetime.now()}][+] Model trained')

        plt.clf()
        plt.title(f'{self.stock.name} ({self.stock.symbol}) - Estimation')
        plt.plot(yhat, label=f'Prediction (RMSE : {rmse_sent})')
        plt.plot(test_y, label='Valeur du marché')
        if extra_indicators is not None:
            plt.plot(extra_indicators)
        plt.legend()
        if save_results:
            plt.savefig(f'{self.dir}/{self.stock.symbol}-plot.png')
        else:
            plt.show()
        tf.reset_default_graph()

    @staticmethod
    def process_data(in_data):
        out_data = []
        for line in in_data:
            out_data.append(float(line.split(',')[0]))
        return np.array(out_data).reshape(-1, 1)
