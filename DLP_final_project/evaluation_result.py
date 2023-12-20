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
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from utiles.dataloader import BIDMC_DataLoader, read_data
from utiles.dataloader import Load_best_model


def caculate_rmse_mse(ecg, fake_ecg):

    diff=np.subtract(ecg, fake_ecg)
    square=np.square(diff)
    MSE=square.mean()
    RMSE=np.sqrt(MSE)

    return RMSE, MSE

def caculate_mae(ecg, fake_ecg):
    return np.mean(np.abs(ecg - fake_ecg))


if __name__ == "__main__": 

    data_dir = './data'
    model_save_dir = "./results"
    model_tag = "WGAN_Generator_CBAM_0.2advloss_nosubjectwise_200epoch"
    save_model_path = model_save_dir + "/" +  model_tag + "/"
    generator_model_path = save_model_path + 'Generator'
    loss_history_path = save_model_path + 'Loss_history.csv'

    if os.path.isdir(save_model_path + 'eval_result'):
        print("The dir is exist!!!")
        print("Press Enter to continue")
    else:
        os.mkdir(model_save_dir + "/" + model_tag + '/eval_result')

    eval_result_path = model_save_dir + "/" + model_tag + '/eval_result/'

    #--------------------------------------------------------------------------------------------------------------
    test_batch_size = 1
    save_fig_flag = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #--------------------------------------------------------------------------------------------------------------
    ## read and all_data from pkl file
    print('Loading all data...')

    test_all_data = read_data(r'C:\Users\Tree\OneDrive - nctu.edu.tw\8Senior\深度學習實驗\DLP_final_project\data\BIDMC_preprocess_30s_ripple_filt_step_5s_ECG12_v2\test')
    test_dataset = BIDMC_DataLoader(test_all_data)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    print('Done')
    #--------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------
    ## Load model
    print('Loading model ...')
    netG_path = Load_best_model(generator_model_path)
    netG = torch.load(generator_model_path + '/' + netG_path).to(device)

    print('Done.')

    # -------------------------------------------------------------------------------------------------------------
    total_mse = []
    total_rmse = []
    total_mae = []
    total_P = []

   # val_count = 0
    avg_eval_acc = []
    with torch.no_grad():
        start_time = datetime.now()
        progress_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        for step, batch in progress_bar:

            PPG = batch['PPG'].unsqueeze(1)
            ECG = batch['ECG'].unsqueeze(1)
            seg_ID = batch['segment_id']

            PPG = PPG.to(device)
            ECG = ECG.to(device)

            # G
            Genh_sig = netG(PPG)
            
            rmse, mse = caculate_rmse_mse(ECG.detach().cpu()[0][0].numpy(), Genh_sig.detach().cpu()[0][0].numpy())
            mae = caculate_mae(ECG.detach().cpu()[0][0].numpy(), Genh_sig.detach().cpu()[0][0].numpy())
            P = np.abs(np.corrcoef(ECG.detach().cpu()[0][0].numpy(), Genh_sig.detach().cpu()[0][0].numpy())[0][1])

            if save_fig_flag:
                title = str(seg_ID[0]) + ' P = ' + str(P)
                fig_name = eval_result_path + seg_ID[0] + '.png'
                fig, ax = plt.subplots(4,1)

                ax[0].plot(PPG.detach().cpu()[0][0].numpy())
                ax[0].axis(ymax=-1, ymin=1)
                ax[0].set_title('PPG')

                ax[1].plot(ECG.detach().cpu()[0][0].numpy())
                ax[1].axis(ymax=-1, ymin=1)
                ax[1].set_title('ECG')

                ax[2].plot(Genh_sig.detach().cpu()[0][0].numpy())
                ax[2].axis(ymax=-1, ymin=1)
                ax[2].set_title('Reconstruct ECG')
                
                ax[3].plot(ECG.detach().cpu()[0][0].numpy(), label='ground truth ECG')
                ax[3].plot(Genh_sig.detach().cpu()[0][0].numpy(), label='Reconstruct ECG')
                ax[3].axis(ymax=-1, ymin=1)
                ax[3].set_title('Compare ECG')
                ax[3].legend()

                fig.set_size_inches(15, 12)

                fig.suptitle(title, fontsize=15)
                plt.tight_layout()
                plt.savefig(fig_name)
                plt.close()

            

            total_mse.append(mse)
            total_rmse.append(rmse)
            total_mae.append(mae)
            total_P.append(P)

            cur_time = datetime.now()
            delta_time = cur_time - start_time
            delta_time = timedelta(seconds=delta_time.seconds) 
            progress_bar.set_description(f"Test: [{step + 1}/{len(test_dataloader)}] "
                                                    # f"Acc: {np.mean(avg_eval_acc):.3f}%"
                                                    f"Time: {delta_time}\33[0m")

        print('------------- Result ------------------')
        print(model_tag)
        print('RMSE = ', np.mean(total_rmse))
        print('MSE = ', np.mean(total_mse))
        print('MAE = ', np.mean(total_mae))
        print('P = ', np.mean(total_P))


        print('Max P = ', np.max(total_P))