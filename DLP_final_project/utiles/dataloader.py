import os
import sys

import torch
import pandas as pd
import numpy as np
from scipy import signal
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def read_data(path):
    all_data = []
    dir_list = os.listdir(path)

    for i in range(len(dir_list)):
        file_name_list = os.listdir(path+ "\\" +dir_list[i])
        data_list = []
        
        for j in range(len(file_name_list)):
            df = pd.read_csv(path+ "\\" +dir_list[i] + "\\" +file_name_list[j])
            PPG_data = np.array(df.PPG)
            ECG_data = np.array(df.ECG)
            data_list.append([file_name_list[j].split('.csv')[0], PPG_data, ECG_data])
            pass
        all_data.append([dir_list[i], data_list])
    return all_data


class BIDMC_DataLoader(Dataset):
    def __init__(self, data):
        ECG_data = list()
        PPG_data = list()
        data_id = list()
        sub_id = list()


        for i in range(len(data)):
            data_num = len(data[i][1])
            for j in range(data_num):
                sub_id.append(data[i][0])
                data_id.append(data[i][1][j][0])
                PPG_data.append(data[i][1][j][1])
                ECG_data.append(data[i][1][j][2])
                pass
            pass

        self.SUB_ID = np.asarray(sub_id, dtype=np.str)
        self.PPG_DATA = np.asarray(PPG_data, dtype=np.float32)
        self.DATA_ID = np.asarray(data_id, dtype=np.str)
        self.ECG_DATA = np.asarray(ECG_data, dtype=np.float32)

        self.PPG_DATA = torch.from_numpy(self.PPG_DATA).float()
        self.ECG_DATA = torch.from_numpy(self.ECG_DATA).float()
        
        
    def __getitem__(self, index):    
        id = self.SUB_ID[index]
        seg_id = self.DATA_ID[index]
        ppg_data = self.PPG_DATA[index] 
        ecg_data= self.ECG_DATA[index]

        entry = {'subject_id': id, 'segment_id': seg_id, 'PPG': ppg_data, 'ECG' : ecg_data}
        return entry
    
    def __len__(self):
        return len(self.PPG_DATA)  


def Load_best_model(path):
    dir_list = os.listdir(path)
    best_model_loss = 1000
    for i in range(len(dir_list)):
        loss = float(dir_list[i].split('loss_')[1].split('.pth')[0])
        if loss < best_model_loss:
            best_model_path = dir_list[i]
            best_model_loss = loss
    
    return best_model_path



if __name__== "__main__": 
    all_data = read_data('././data/BIDMC_preprocess_20s/test')
    bidmc_dataset = BIDMC_DataLoader(all_data)
    train_dataloader = DataLoader(bidmc_dataset, batch_size=128, shuffle=True)

    from tqdm import tqdm
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for step, batch in progress_bar:
        batch['PPG']
        batch['ECG']

    # -------------------------------------------------------------------------------------
    # best_model_path = Load_best_model('././results/SEGAN_dual_discriminator_adam_epoch600_lr1e4/Generator')
    # print(best_model_path)

    pass
