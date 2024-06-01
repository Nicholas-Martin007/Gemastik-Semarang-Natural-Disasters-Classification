import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

import torch
from model import LSTM_model, createLinearRegression
from data_loader import clean_all_data, create_time_series_dataset, get_data_loader, split_data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# from kneed import KneeLocator

from test import test_model
from train import train_model

def train_pipeline(data, train_test_split, regression_models='linear_regression.pkl'):
   data_train = data[data.index <= train_test_split]
   data_test = data[data.index > train_test_split]

   # data_train = data[data.Tanggal <= train_test_split]
   # data_test = data[data.Tanggal > train_test_split]


   X_train, y_train = data_train.drop(['B', 'RB', 'TL', 'KB', 'PT'], axis=1), data_train[['B', 'RB', 'TL', 'KB', 'PT']]
   X_test, y_test = data_test.drop(['B', 'RB', 'TL', 'KB', 'PT'], axis=1), data_test[['B', 'RB', 'TL', 'KB', 'PT']]


   # data_train = data_train.groupby(['Tanggal', 'Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x',
   #    'ff_avg', 'KECAMATAN_Banyumanik',
   #    'KECAMATAN_Candisari', 'KECAMATAN_Gajah Mungkur', 'KECAMATAN_Gayamsari',
   #    'KECAMATAN_Genuk', 'KECAMATAN_Gunungpati', 'KECAMATAN_Mranggen',
   #    'KECAMATAN_Ngaliyan', 'KECAMATAN_Pedurungan',
   #    'KECAMATAN_Semarang Barat', 'KECAMATAN_Semarang Selatan',
   #    'KECAMATAN_Semarang Tengah', 'KECAMATAN_Semarang Timur',
   #    'KECAMATAN_Semarang Utara', 'KECAMATAN_Tembalang', 'KECAMATAN_Tugu',
   #    'ddd_car_C ', 'ddd_car_E ', 'ddd_car_N ', 'ddd_car_NW', 'ddd_car_S ',
   #    'ddd_car_SE', 'ddd_car_SW', 'ddd_car_W '], as_index = False).sum()

   # ===============
   # data_train = data_train[['Tanggal', 'Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x',
   #    'ff_avg', 'ddd_car_C ', 'ddd_car_E ', 'ddd_car_N ', 'ddd_car_NW', 'ddd_car_S ',
   #    'ddd_car_SE', 'ddd_car_SW', 'ddd_car_W ', 'B', 'RB', 'TL', 'KB', 'PT']].groupby(['Tanggal', 'Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x',
   #    'ff_avg', 'ddd_car_C ', 'ddd_car_E ', 'ddd_car_N ', 'ddd_car_NW', 'ddd_car_S ',
   #    'ddd_car_SE', 'ddd_car_SW', 'ddd_car_W '], as_index = False).sum()
   # data_train["Total Bencana Alam"] = data_train[['B', 'RB', 'TL', 'KB', 'PT']].sum(axis = 1)
   # data_train = data_train.drop(['B', 'RB', 'TL', 'KB', 'PT'], axis = 1)
   # data_train.set_index('Tanggal', inplace = True)

   # data_test = data_test[['Tanggal', 'Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x',
   #    'ff_avg', 'ddd_car_C ', 'ddd_car_E ', 'ddd_car_N ', 'ddd_car_NW', 'ddd_car_S ',
   #    'ddd_car_SE', 'ddd_car_SW', 'ddd_car_W ', 'B', 'RB', 'TL', 'KB', 'PT']].groupby(['Tanggal', 'Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x',
   #    'ff_avg', 'ddd_car_C ', 'ddd_car_E ', 'ddd_car_N ', 'ddd_car_NW', 'ddd_car_S ',
   #    'ddd_car_SE', 'ddd_car_SW', 'ddd_car_W '], as_index = False).sum()
   # data_test["Total Bencana Alam"] = data_test[['B', 'RB', 'TL', 'KB', 'PT']].sum(axis = 1)
   # data_test = data_test.drop(['B', 'RB', 'TL', 'KB', 'PT'], axis = 1)
   # data_test.set_index('Tanggal', inplace = True)


   # X_train, y_train = data_train.drop(['Total Bencana Alam'], axis=1), data_train[['Total Bencana Alam']]
   # X_test, y_test = data_test.drop(['Total Bencana Alam'], axis=1), data_test[['Total Bencana Alam']]



   createLinearRegression(X_train, y_train, X_test, y_test)
   model = pickle.load(open('pkl_models/' + regression_models, 'rb'))
   

   return 

if __name__ == '__main__':
   # B	   : BANJIR
   # RB     : ROB
   # TL	   : TANAH LONGSOR
   # PT	   : POHON TUMBANG
   # KB	   : KEBAKARAN

   # clean_all_data()

   # Read data
   # data = pd.read_csv(f'./data_kecamatan_clean/data_Semarang Barat.csv', index_col=0)
   data = pd.read_csv('data_combine.csv', index_col=0)

   # # ====================================
   # # PCA (Only inputs(!Tanggal, !Bencana Alam))
   # scaler = StandardScaler()
   # data_scaled = scaler.fit_transform(data.iloc[:, 0:data.shape[1]-1].astype(float))
   # # print(data_scaled)
   
   # df_scaled = pd.DataFrame(data=data_scaled, columns=data.iloc[:, 0:data.shape[1]-1].columns)
   # # print(df_scaled)

   # pca = PCA(n_components=df_scaled.shape[1])
   # transform_pca = pca.fit_transform(df_scaled)
   
   # # Cummulative proportion
   # pk = pca.explained_variance_ratio_
   # print(pk)

   # pca_number = np.arange(pca.n_components_) + 1
   # print(pca_number)

   # variance = pca.explained_variance_

   # # Elbow Method
   # plt.figure(figsize=(10, 6))
   # plt.plot(pca_number, pk, marker='o')
   # plt.xlabel('Component Number')
   # plt.ylabel('Proporsi Kumulatif') # can be PK, but can't use axhline
   # # plt.axhline(y=1, color='r', linestyle='--')
   # plt.show()

   # # elbow_locator = KneeLocator(x=pca_number, y=pk, direction="decrease") # error
   # # elbow_point = elbow_locator.knee - 1
   # # print(elbow_point)

   # # got the best component is 9
   # pca = PCA(n_components=9)
   # transform_pca = pca.fit_transform(df_scaled)
   # pca_number = len(np.arange(pca.n_components_) + 1)

   # # pc1, pc2, pc3, pc4, pc5, pc6, pc7, pc8, pc9 = [transform_pca[:, i] for i in range(pca_number)]
   # # ====================================


   for features in ['Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg']:
      data[features] = StandardScaler().fit_transform(data[[features]])
   
   # Concat to be time series input and output
   data = data[list(data)[0:data.shape[1]]].astype(float)
   time_series_data = create_time_series_dataset(data, past_steps=14)

   val_size = 0.4
   test_size = 0.2
   batch_size = 32
   train_loader, val_loader, test_loader = get_data_loader(time_series_data, batch_size, val_size, test_size)

   # model = LSTM_model(input_size=18, hidden_size=64, num_layers=2, num_classes=len(data['Bencana Alam'].unique()))
   model = LSTM_model(input_size=18, hidden_size=64, num_layers=2, num_classes=1)

   train_ = 1
   
   if train_:
      train_model(model, batch_size, train_loader, val_loader, lr=0.001, epochs=100)

   
   device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
   model_name = "model_LSTM_model_batch_size32_lr_0.001_epoch_100"
   model.load_state_dict(torch.load(model_name, map_location=device))

   test_model(model, test_loader)
   print("DONE")

   

   # https://discuss.pytorch.org/t/runtimeerror-multi-target-not-supported-newbie/10216
   # https://discuss.pytorch.org/t/why-does-nn-embedding-layers-expect-longtensor-type-input-tensors/21376/2

   # df = pd.read_csv('./data_bencana_semarang_clean/data_all_years.csv')
   # df = pd.read_csv('./data_bencana_semarang_clean/data_all_years.csv',index_col='Tanggal')
   
   # train_pipeline(data=df, train_test_split='2021-05-01', regression_models='linear_regression.pkl')
