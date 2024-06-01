import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

import re

import torch
from torch.utils.data import DataLoader

hashmap = {
    'Candisari': 'Candisari',#
    'Semarang Barat': 'Semarang Barat',#
    'Banyumanik': 'Banyumanik',#
    'Gunung Pati': 'Gunungpati',
    'Semarang Timur': 'Semarang Timur',#
    'Semarang selatan': 'Semarang Selatan',#
    'Gajah Mungkur': 'Gajah Mungkur',#
    'Tugu': 'Tugu',#
    'Tembalang': 'Tembalang',#
    'T u g u': 'Tugu',
    'Ngaliyan': 'Ngaliyan',#
    'Gunungpati': 'Gunungpati',#
    'Genuk': 'Genuk',#
    'Gajah mungkur': 'Gajah Mungkur',
    'Gayamsari': 'Gayamsari',#
    'Gajahmungkur': 'Gajah Mungkur',
    'Semarang Sltn': 'Semarang Selatan',
    'Gn Pati': 'Gunungpati',
    'Semarang Tmr': 'Semarang Timur',
    'Pedurungan': 'Pedurungan',
    'Semarang Brt': 'Semarang Barat',
    'Semarang Utara': 'Semarang Utara',#
    'Gn. Pati': 'Gunungpati',
    'Semarang Barat ': 'Semarang Barat',
    'Tembalang ': 'Tembalang',
    'Smg Barat': 'Semarang Barat',
    'Semarang Tgh.': 'Semarang Tengah',
    'G e n u k': 'Genuk',
    'Semarang Tgh': 'Semarang Tengah',
    'Mranggen': 'Mranggen',#
    'Kab. Demak': 'Kabupaten Demak',
    'Gnpati': 'Gunungpati',
    'Smg Timur': 'Semarang Timur',
    'Kec Sayung': 'Kecamatan Sayung',
    'Kab Demak': 'Kabupaten Demak',
    'Smg Utara': 'Semarang Utara',
    'Pedurungan ': 'Pedurungan',#
    'Smg Tengah': 'Semarang Tengah',#
    'Pedurungan  Lor': 'Pedurungan',
    'Smg Berat': 'Semarang Barat',
    'Semarang Tengah': 'Semarang Tengah',
    'Semarang Sltn.': 'Semarang Selatan',
    'Semarang Utara ': 'Semarang Utara',
    'T u g u ': 'Tugu',
    
    "Semarang Tngh": "Semarang Tengah",
    "Semarang Utr": "Semarang Utara",

    "Pedurungan Kidul": "Pedurungan",
    "Sukorejo": "Gunungpati",
    "Srondol Kulon": "Banyumanik",
    "Bangunharjo": "Banyumanik",
    "Meteseh": "Tembalang",
}

month_map = {
    'Januari': 'January',
    'Februari': 'February',
    'Maret': 'March',
    'April': 'April',
    'Mei': 'May',
    'Juni': 'June',
    'Juli': 'July',
    'Agustus': 'August',
    'September': 'September',
    'Oktober': 'October',
    'November': 'November',
    'Desember': 'December'
}



def clean_bmkg(year):
    data_bmkg_januari = pd.read_excel(f'./Data Cuaca Harian BMKG/Data {year}/Januari.xlsx', header=8)
    data_bmkg_februari = pd.read_excel(f'./Data Cuaca Harian BMKG/Data {year}/Februari.xlsx', header=8)
    data_bmkg_maret = pd.read_excel(f'./Data Cuaca Harian BMKG/Data {year}/Maret.xlsx', header=8)
    data_bmkg_april = pd.read_excel(f'./Data Cuaca Harian BMKG/Data {year}/April.xlsx', header=8)
    data_bmkg_mei = pd.read_excel(f'./Data Cuaca Harian BMKG/Data {year}/Mei.xlsx', header=8)
    data_bmkg_juni = pd.read_excel(f'./Data Cuaca Harian BMKG/Data {year}/Juni.xlsx', header=8)
    data_bmkg_juli = pd.read_excel(f'./Data Cuaca Harian BMKG/Data {year}/Juli.xlsx', header=8)
    data_bmkg_agustus = pd.read_excel(f'./Data Cuaca Harian BMKG/Data {year}/Agustus.xlsx', header=8)
    data_bmkg_september = pd.read_excel(f'./Data Cuaca Harian BMKG/Data {year}/September.xlsx', header=8)
    data_bmkg_oktober = pd.read_excel(f'./Data Cuaca Harian BMKG/Data {year}/Oktober.xlsx', header=8)
    data_bmkg_november = pd.read_excel(f'./Data Cuaca Harian BMKG/Data {year}/November.xlsx', header=8)
    data_bmkg_desember = pd.read_excel(f'./Data Cuaca Harian BMKG/Data {year}/Desember.xlsx', header=8)

    data_bmkg = pd.concat([data_bmkg_januari, data_bmkg_februari, data_bmkg_maret,
                                data_bmkg_april, data_bmkg_mei, data_bmkg_juni,
                                data_bmkg_juli, data_bmkg_agustus, data_bmkg_september,
                                data_bmkg_oktober, data_bmkg_november, data_bmkg_desember])

    data_bmkg['Tanggal'] = pd.to_datetime(data_bmkg['Tanggal'], format='%d-%m-%Y', errors='coerce')
    data_bmkg = data_bmkg.dropna(subset=['Tanggal'])

    data_bmkg['Tn'].fillna(round(data_bmkg['Tn'].mean(), 1), inplace=True)
    data_bmkg['Tx'].fillna(round(data_bmkg['Tx'].mean(), 1), inplace=True)
    data_bmkg['Tavg'].fillna(round(data_bmkg['Tavg'].mean(), 1), inplace=True)
    data_bmkg['RH_avg'].fillna(round(data_bmkg['RH_avg'].mean(), 1), inplace=True)
    data_bmkg['RR'].fillna(0, inplace=True)
    data_bmkg['RR'].replace(8888, 0, inplace=True)
    data_bmkg['ss'].fillna(0, inplace=True)
    data_bmkg['ddd_x'].fillna(round(data_bmkg['ddd_x'].mean(), 1), inplace=True)
    data_bmkg['ff_x'].fillna(round(data_bmkg['ff_x'].mean(), 1), inplace=True)
    data_bmkg['ff_avg'].fillna(round(data_bmkg['ff_avg'].mean(), 1), inplace=True)
    data_bmkg['ddd_car'].fillna('C', inplace=True)

    data_bmkg['Tanggal'] = pd.to_datetime(data_bmkg['Tanggal'], format='%d-%m-%Y')

    data_bmkg = data_bmkg.dropna(subset=['Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg', 'ddd_car'], how='all')

    data_bmkg.to_csv(f'./data_bencana_semarang_clean/data_bmkg_{year}.csv', index=False)





def clean_data(filename, header):
    
    data_semarang = pd.read_excel(filename, header=header)

    # ===== Drop unnecessary =====
    drop_columns = ['NO.', 'PB', 'RR', 'MD', 'Luka2', 'HLG', 'KERUGIAN', 'KETERANGAN']
    if filename != './data_bencana_semarang/DATA_BENCANA_2019.xlsx':
        drop_columns += ['Korban ', 'Pengungsi']
    data_semarang.drop(columns = drop_columns, inplace=True)
    # =====

    # Rename features column
    data_semarang.rename(columns={"TGL. KEJADIAN": "TGL KEJADIAN", "B ": "B"}, inplace=True)

    # ===== Remove unnecessary =====
    data_semarang_remove_date = ['TGL. KEJADIAN', '01 - 17 Februari 2018', '01-05 Maret 2018', 
                                '14/04/2018', '29 - 30 Januari 2018', '01-02 Maret 2018', 
                                '   : BANJIR', '   : ROB', '   : TANAH LONGSOR', 
                                '   : PUTING BELIUNG', '   : RUMAH ROBOH', '   : POHON TUMBANG', 
                                '   : KEBAKARAN', ': Meninggal Dnia', ': Luka2', 
                                ':Hilang', '   : TNH LONGSOR',
                                # data_semarang_2
                                'Hasil kegiatan orientasi dari pkl. 18.00 wib s/d pkl 22.00 wib :',
                                'wib :',
                                'TGL. KEJADIAN',
                                '(9/6/2019)',
                                # data_semarang_3
                                'S . D . A ',
                                '( 03/10/2020 )',
                                # data_semarang_4
                                "( Jum'at )",
                                '( Sabtu )'
                                ]
    data_semarang = data_semarang[~data_semarang['TGL KEJADIAN'].isin(data_semarang_remove_date)]
    # =====

    # Remove total kejadian, jumlah, ... that doesn't have LOKASI and KECAMATAN
    data_semarang.dropna(subset=['LOKASI', 'KECAMATAN'], how='all', inplace=True)

    # ===== Fixing EXCEL date format problem, for example: 44912, 44920 =====
    if filename == './data_bencana_semarang/DATA_BENCANA_2021.xlsx' or filename == './data_bencana_semarang/DATA_BENCANA_2022.xlsx':
        data_semarang['TGL KEJADIAN'] = pd.TimedeltaIndex(data_semarang['TGL KEJADIAN'], unit='d') + datetime(1899,12,30)
    # ====

    # ===== Fill nan value for labels ====
    columns_to_fill = ['B', 'RB', 'KB', 'TL', 'PT']
    data_semarang[columns_to_fill] = data_semarang[columns_to_fill].fillna(0)
    # =====

    # ===== Rename filename =====
    if filename == './data_bencana_semarang/DATA_BENCANA_2020.xlsx':
        data_semarang.loc[data_semarang['TGL KEJADIAN'] == '19/02/2020 s / d', 'TGL KEJADIAN'] = '19/02/2020'
        data_semarang.loc[data_semarang['TGL KEJADIAN'] == '29/02.2020', 'TGL KEJADIAN'] = '20/02/2020'
    # =====

    # Remove unnecessary data
    data_semarang = data_semarang.dropna(subset=['TGL KEJADIAN', 'LOKASI', 'KELURAHAN'], how='all')


    # ===== Format Date, add up labels each date, fill up date if the data has labels
    previous_date = None
    date_format_regex = r'^\d{2} (Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember) \d{4}$'
    index = 0
    B_temp, RB_temp, TL_temp, KB_temp, PT_temp = 0, 0, 0, 0, 0

    for i, data in data_semarang.iterrows():
        # ===== Reformat Date =====
        data_tanggal = str(data['TGL KEJADIAN'])
        if re.match(date_format_regex, data_tanggal):

            for indo_month, eng_month in month_map.items():
                data_tanggal = data_tanggal.replace(indo_month, eng_month)

            data_semarang.at[i, 'TGL KEJADIAN'] = datetime.strptime(data_tanggal, '%d %B %Y')
        # =====

        if pd.notnull(data['TGL KEJADIAN']):
            if (B_temp + RB_temp + TL_temp + KB_temp + PT_temp) >= 1:
                data_semarang.at[index, 'B'] =  int(data_semarang.at[index, 'B']) + B_temp
                data_semarang.at[index, 'RB'] = int(data_semarang.at[index, 'RB']) + RB_temp
                data_semarang.at[index, 'TL'] = int(data_semarang.at[index, 'TL']) + TL_temp
                data_semarang.at[index, 'KB'] = int(data_semarang.at[index, 'KB']) + KB_temp
                data_semarang.at[index, 'PT'] = int(data_semarang.at[index, 'PT']) + PT_temp

            index = i
            B_temp, RB_temp, TL_temp, KB_temp, PT_temp = 0, 0, 0, 0, 0
        else:
            if pd.notnull(data['B']): B_temp += data['B']
            if pd.notnull(data['RB']): RB_temp += data['RB']
            if pd.notnull(data['TL']): TL_temp += data['TL']
            if pd.notnull(data['KB']): KB_temp += data['KB']
            if pd.notnull(data['PT']): PT_temp += data['PT']

        # get empty date == prev date
        if pd.isnull(data['TGL KEJADIAN']):
            if previous_date and data['KECAMATAN'] != previous_kecamatan:
                data_semarang.at[i, 'TGL KEJADIAN'] = previous_date

        else:
            previous_date = data['TGL KEJADIAN']
            previous_kecamatan = data['KECAMATAN']

        # TYPO FIX
        data_semarang.at[i, 'KECAMATAN'] = hashmap.get(data['KECAMATAN'], data['KECAMATAN'])

    # =====

    # ===== Fixing Weird date format =====
    def convert_to_datetime(date):
        if isinstance(date, datetime):
            return date
        elif pd.isna(date):
            return
        return datetime.strptime(date, '%d/%m/%Y')

    data_semarang['TGL KEJADIAN'] = [convert_to_datetime(date) for date in data_semarang['TGL KEJADIAN']]
    # =====

    # ===== Remove unused data['KECAMATAN'], drop column of LOKASI + KELURAHAN =====
    data_semarang.dropna(subset=['KECAMATAN'], inplace=True)
    data_semarang = data_semarang[~(data_semarang[['B', 'RB', 'TL', 'KB', 'PT']] == 0).all(axis=1)]
    data_semarang.drop(columns=['KELURAHAN', 'LOKASI'], inplace=True)
    data_semarang.dropna(subset=['TGL KEJADIAN'], inplace=True)
    # ======

    # ===== Match semarang kecamatan ====
    kecamatan_semarang = ['Banyumanik', 'Gajah Mungkur', 'Candisari', 'Ngaliyan', 'Gunungpati', 'Mijen', 'Semarang Timur', 'Pedurungan', 'Semarang Barat', 'Semarang Tengah', 'Tugu', 'Tembalang', 'Gayamsari', 'Semarang Selatan', 'Semarang Utara', 'Genuk']
    data_semarang = data_semarang[data_semarang['KECAMATAN'].isin(kecamatan_semarang)]
    # =====

    #Typo Fix
    data_semarang.loc[data_semarang['TL'] == '`1', 'TL'] = 1

    # ===== Get all kecamatan at each date =====
    date_ranges = {
        './data_bencana_semarang/DATA_BENCANA_2019.xlsx': ('2019-01-01', '2019-12-31'),
        './data_bencana_semarang/DATA_BENCANA_2020.xlsx': ('2020-01-01', '2020-12-31'),
        './data_bencana_semarang/DATA_BENCANA_2021.xlsx': ('2021-01-01', '2021-12-31'),
        './data_bencana_semarang/DATA_BENCANA_2022.xlsx': ('2022-01-01', '2022-12-31'),
    }
    if filename in date_ranges:
        start, end = date_ranges[filename]
        
    dates_year = pd.date_range(start=start, end=end, freq='D')
    unique_kecamatan = pd.DataFrame({'KECAMATAN': data_semarang['KECAMATAN'].unique()})
    cartesian_product = pd.DataFrame([(date, kecamatan) for date in dates_year for kecamatan in unique_kecamatan['KECAMATAN']], columns=['TGL KEJADIAN', 'KECAMATAN'])
    data_semarang['TGL KEJADIAN'] = pd.to_datetime(data_semarang['TGL KEJADIAN'])
    data_semarang = pd.merge(cartesian_product, data_semarang, on=['TGL KEJADIAN', 'KECAMATAN'], how='left').fillna(0)
    data_semarang = data_semarang.groupby(['TGL KEJADIAN', 'KECAMATAN']).sum().reset_index()
    # =====

    # ===== FOR WEIRD DATE FORMAT:
    def swap_dates(date):
        if 1 <= date.day <= 12:
            swapped_date = pd.Timestamp(date.year, date.day, date.month)
            return swapped_date
        else:
            return date  # Return unchanged date if format is incorrect

    for i, date in enumerate(data_semarang['TGL KEJADIAN']):
        data_semarang.at[i, 'TGL KEJADIAN'] = swap_dates(date)

    data_semarang.sort_values(by='TGL KEJADIAN', inplace=True)
    # =====

    data_semarang[['B', 'RB', 'TL', 'KB', 'PT']] = data_semarang[['B', 'RB', 'TL', 'KB', 'PT']].applymap(lambda x: 1 if x >= 1 else x)

    # ===== Save File =====
    file_name = {
        './data_bencana_semarang/DATA_BENCANA_2019.xlsx': '2019',
        './data_bencana_semarang/DATA_BENCANA_2020.xlsx': '2020',
        './data_bencana_semarang/DATA_BENCANA_2021.xlsx': '2021',
        './data_bencana_semarang/DATA_BENCANA_2022.xlsx': '2022',
    }

    if filename in file_name:
        output_path = f'./data_bencana_semarang_clean/data_semarang_{file_name[filename]}_clean.csv'
        data_semarang.to_csv(output_path, index=False)
    # =====

def clean_all_data():
    # ================================================================================
    clean_bmkg('2019')
    clean_bmkg('2020')
    clean_bmkg('2021')
    clean_bmkg('2022')

    clean_data(filename='./data_bencana_semarang/DATA_BENCANA_2019.xlsx', header=4)
    clean_data(filename='./data_bencana_semarang/DATA_BENCANA_2020.xlsx', header=5)
    clean_data(filename='./data_bencana_semarang/DATA_BENCANA_2021.xlsx', header=5)
    clean_data(filename='./data_bencana_semarang/DATA_BENCANA_2022.xlsx', header=5)

    # ===== Concat data_bmkg + data_semarang
    def concat_data_bmkg_semarang(year):
       data_bmkg = pd.read_csv(f'./data_bencana_semarang_clean/data_bmkg_{year}.csv')
       data_semarang = pd.read_csv(f'./data_bencana_semarang_clean/data_semarang_{year}_clean.csv')

       merge_data = pd.merge(data_bmkg, data_semarang, left_on='Tanggal', right_on='TGL KEJADIAN', how='right')
       merge_data.drop(columns=['TGL KEJADIAN'], inplace=True)
       merge_data.to_csv(f'./data_bencana_semarang_clean/merge_data_{year}.csv', index=False)

       return merge_data


    years = ['2019', '2020', '2021', '2022']
    total = [concat_data_bmkg_semarang(year) for year in years]

    data_all_years = pd.concat(total)
    # data_all_years = pd.get_dummies(data_all_years, columns=['KECAMATAN', 'ddd_car'], dtype=float)
    data_all_years.to_csv('./data_bencana_semarang_clean/data_all_years.csv', index=False)

    data = pd.read_csv('./data_bencana_semarang_clean/data_all_years.csv')
    data['Bencana Alam'] = data[['B', 'RB', 'TL', 'KB', 'PT']].astype(int).astype(str).sum(axis=1)
    # data['Bencana Alam'] = data['Bencana Alam'].apply(lambda x: int(x, 2))
    data.to_csv('./data_bencana_semarang_clean/data_all_years.csv', index=False)
    data.drop(columns=['B', 'RB', 'TL', 'PT', 'KB'], inplace=True)

    data_all = None
    i = 0

    for kecamatan in data['KECAMATAN'].unique():
        data_kecamatan = data[data['KECAMATAN'] == kecamatan]
        data_kecamatan = pd.get_dummies(data_kecamatan, columns=['ddd_car'], dtype=float, drop_first=True)
            
        data_kecamatan.drop(columns=['KECAMATAN'], inplace=True)

        # Make output at last column index
        bencana_alam = data_kecamatan.pop('Bencana Alam')
        data_kecamatan['Bencana Alam'] = bencana_alam
            
        data_kecamatan.to_csv(f'./data_kecamatan_clean/data_{kecamatan}.csv', index=False)

        if i == 0:
            data_all = data_kecamatan
            i += 1
        else:
            data_all = pd.concat([data_all, data_kecamatan], axis=0)

        

        # ===== For spliting all output (NOT USED)
        # B, RB, TL, KB, PT = data_kecamatan.pop('B'), data_kecamatan.pop('RB'), data_kecamatan.pop('TL'), data_kecamatan.pop('KB'), data_kecamatan.pop('PT')
        # bencana_alam = ['B', 'RB', 'TL', 'PT', 'KB']

        # for bencana, value in zip(bencana_alam, [B, RB, TL, PT, KB]):
        #     data_kecamatan[bencana] = value
        #     data_kecamatan.to_csv(f'./data_kecamatan_clean/data_{kecamatan}_{bencana}.csv', index=False)
        #     data_kecamatan.pop(bencana)
        # =====

    data_all.to_csv('data_combine.csv', index= False)

def create_time_series_dataset(data, past_steps):
    dataset = []

    for i in range(data.shape[0] - past_steps):
        X = data.iloc[i:i+past_steps, 0:(data.shape[1]-1)].values # feature
        y = data.iloc[i+past_steps, data.shape[1]-1:].values # label

        dataset.append((torch.from_numpy(X), torch.from_numpy(y)))
    
    # print(dataset[0][0].shape)
    # print(dataset[0][1].shape)

    return dataset
    # print(f'X length: {len(X[0:1][0])} : {X[0:1]}')
    # print(f'y length: {len(y[0:1])} : {y[0:1]}')
    
    
def split_data(data, val_size, test_size):

    val_split = int(val_size * len(data))
    test_split = int(test_size * len(data))

    train_data, val_temp = data[val_split:], data[:val_split]
    val_data, test_data = val_temp[test_split:], val_temp[:test_split]

    return train_data, val_data, test_data

def get_data_loader(data, batch_size, val_size, test_size):

    train_data, val_data, test_data = split_data(data, val_size, test_size)

    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    pickle.dump(train_loader, open('train_loader.pkl', 'wb'))
    pickle.dump(val_loader, open('val_loader.pkl', 'wb'))
    pickle.dump(test_loader, open('test_loader.pkl', 'wb'))

    return train_loader, val_loader, test_loader




















