import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
import numpy as np
import os
import re
import pickle
import torch.nn.functional as F
from utiles.dataloader import BIDMC_DataLoader, read_data
# from model import *


if __name__ == "__main__": 
               
    # file_list_path = "file_list_452sub_1.txt"
    data_dir = './data'
    model_save_dir = "./result"
    model_tag = "Unet_DDPM_adam_new_ucnet_200"
    save_model_path = model_save_dir + "/" +  model_tag + "/"
    best_generate_model_path = save_model_path + 'best_generater_model_Adam.pth'
    best_discriminator_model_path = save_model_path + 'best_discriminator_model_Adam.pth'


    if os.path.isdir(model_save_dir + "/" + model_tag):
        print("The dir is exist!!!")
        print("Press Enter to continue")
        input()
    else:
        os.mkdir(model_save_dir + "/" + model_tag)
    log_path = model_save_dir + "/" + model_tag


    #--------------------------------------------------------------------------------------------------------------
    lr = 1e-4
    epoch_num = 200 # 150 -> 100
    train_batch_size = 128
    test_batch_size = 1
    train_only = False
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #--------------------------------------------------------------------------------------------------------------
    ## read and all_data from pkl file
    print('Loading all data...')
    all_data = read_data('C:\chihwei\class\project\data\BIDMC_preprocess')
    bidmc_train_dataset = BIDMC_DataLoader(all_data)
    train_dataloader = DataLoader(bidmc_train_dataset, batch_size=train_batch_size, shuffle=True)

    test_dataset = BIDMC_DataLoader(all_data)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    print('Done')
    #--------------------------------------------------------------------------------------------------------------
    ## Load model
    print('Loading model ...')
    


    # model_optim = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-7)
    # model_optim = optim.SGD(model.parameters(), lr=0.001)
    print('Done.')

    # -------------------------------------------------------------------------------------------------------------
    ## Start training
    train_loss_history = []
    test_loss_history = []

    best_eval_acc = 0
    print('Start training ...')
    for epoch in range(epoch_num):
        
        total_loss = 0
        start_time = datetime.now()
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for step, batch in progress_bar:
            PPG = batch['PPG']
            ECG = batch['ECG']

            PPG = PPG.to(device)
            ECG = ECG.to(device)
            
            cur_time = datetime.now()
            delta_time = cur_time - start_time
            delta_time = timedelta(seconds=delta_time.seconds) 

            progress_bar.set_description(f"Train: [{epoch + 1}/{epoch_num}][{step + 1}/{len(train_dataloader)}] "
                                                        # f"Loss: {loss.item():.6f} "
                                                        # f"Acc: {np.mean(avg_eval_acc):.3f}%"
                                                        f"Time: {delta_time}\33[0m")
        del total_loss


        avg_eval_acc = []
        with torch.no_grad():
            start_time = datetime.now()
            progress_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
            for step, batch in progress_bar:
                PPG = batch['PPG']
                ECG = batch['ECG']

                PPG = PPG.to(device)
                ECG = ECG.to(device)

                cur_time = datetime.now()
                delta_time = cur_time - start_time
                delta_time = timedelta(seconds=delta_time.seconds) 
                progress_bar.set_description(f"Test: [{epoch + 1}/{epoch_num}][{step + 1}/{len(test_dataloader)}] "
                                                            # f"Loss: {loss.item():.6f} "
                                                            # f"Acc: {np.mean(avg_eval_acc):.3f}%"
                                                            f"Time: {delta_time}\33[0m")

    

    # ======================= save AUC ================================
    df_loss = pd.DataFrame({#'train_Acc': train_ACC_history,
                            'test_Acc': eval_ACC_history})
    
    df_loss.to_csv(save_AUC_history_file_path)
    
    # =================================================================