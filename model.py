import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from pickle import dump

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

import time, sys, os

def createLinearRegression(X_train, y_train, X_test, y_test, show_performance=True):
    model = LinearRegression().fit(X_train, y_train)

    y_fit = pd.DataFrame(model.predict(X_train), index=X_train.index, columns=['B', 'RB', 'TL', 'KB', 'PT'])
    y_pred = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=['B', 'RB', 'TL', 'KB', 'PT'])
    # y_fit = pd.DataFrame(model.predict(X_train), index=X_train.index, columns=['Total Bencana Alam'])
    # y_pred = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=['Total Bencana Alam'])

    train_mse = mean_squared_error(y_train, y_fit)
    test_mse = mean_squared_error(y_test, y_pred)

    if show_performance:
        print("Train score: ", train_mse)
        print("Test score: ", test_mse)
    
    # ax = y_pred.plot(y='Total Bencana Alam', label='Predicted')
    # ax = y_test.plot(y='Total Bencana Alam', ax=ax, label='Real Data')
    # plt.show()

    # fig, ax = plt.subplots()

    # ax.scatter(y_pred.index, y_pred['Total Bencana Alam'], label='Predicted', color='blue')
    # ax.scatter(y_test.index, y_test['Total Bencana Alam'], label='Real Data', color='red')
    # ax.legend()
    # plt.show()

    ax = y_pred.plot(y=['B', 'RB', 'TL', 'KB', 'PT'])
    ax = y_test.plot(y=['B', 'RB', 'TL', 'KB', 'PT'], ax=ax)
    plt.show()

    

    dump(model, open('pkl_models/linear_regression.pkl', 'wb'))


class LSTM_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM_model, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.name = "LSTM_model"

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2).double()
        self.fc = nn.Linear(hidden_size, num_classes).double()

    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        hidden_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device).double()
        cell_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device).double()

        out, _ = self.rnn(x, (hidden_state)) # 16 kurang tau dari mana

        out = out[:, -1, :]

        out = self.fc(out)

        return out
