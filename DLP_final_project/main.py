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
import torch.nn.functional as F
from utiles.dataloader import BIDMC_DataLoader, read_data
from nets.discriminator import pairDiscriminator, WGAN_pairDiscriminator
from nets.generator import SeGAN_Generator
from nets.generator import SeGAN_Generator_CBAM_easy_control
from utiles.design_loss_fun import ReconstructLoss
# from model import *


if __name__ == "__main__": 

    data_dir = './data'
    model_save_dir = "./results"
    model_tag = "test"
    save_model_path = model_save_dir + "/" +  model_tag + "/"
    loss_history_path = save_model_path + 'Loss_history.csv'

    if os.path.isdir(model_save_dir + "/" + model_tag):
        print("The dir is exist!!!")
        print("Press Enter to continue")
        input()
    else:
        os.mkdir(model_save_dir + "/" + model_tag)
        os.mkdir(model_save_dir + "/" + model_tag + '/Generator')
        os.mkdir(model_save_dir + "/" + model_tag + '/Dis_freq')
        os.mkdir(model_save_dir + "/" + model_tag + '/Dis_sig')


    #--------------------------------------------------------------------------------------------------------------
    lr = 1e-2
    epoch_num = 200 # 150 -> 100
    train_batch_size = 128
    test_batch_size = 1
    train_only = True
    beta1 = 0.9
    beta2 = 0.999
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #--------------------------------------------------------------------------------------------------------------
    ## read and all_data from pkl file
    print('Loading all data...')
    train_all_data = read_data(r'C:\Users\Tree\OneDrive - nctu.edu.tw\8Senior\深度學習實驗\DLP_final_project\data\BIDMC_preprocess_30s_ripple_filt_step_5s_ECG12_v2\train')
    train_dataset = BIDMC_DataLoader(train_all_data)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    test_all_data = read_data(r'C:\Users\Tree\OneDrive - nctu.edu.tw\8Senior\深度學習實驗\DLP_final_project\data\BIDMC_preprocess_30s_ripple_filt_step_5s_ECG12_v2\test')
    test_dataset = BIDMC_DataLoader(test_all_data)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    print('Done')
    #--------------------------------------------------------------------------------------------------------------
    ## Load model
    print('Loading model ...')
    netD_signal = WGAN_pairDiscriminator().to(device)
    netD_signal.init_weights()

    netD_freq = WGAN_pairDiscriminator().to(device)
    netD_freq.init_weights()

    # netG = SeGAN_Generator_CBAM().to(device)
    # netG = SeGAN_Generator_20s_att().to(device)
    # netG = SeGAN_Generator_20s().to(device)
    # netG = SeGAN_Generator_CBAM_stn().to(device)
    # netG = SeGAN_Generator_10s().to(device)
    # netG = SeGAN_Generator_CBAM_easy_control(kernel_size=31, dilation=2).to(device)
    netG = SeGAN_Generator_CBAM_easy_control(kernel_size=31, dilation=2).to(device)

    netG.init_weights()

    del train_dataset, test_dataset

    # optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=1e-08, weight_decay=1e-7)
    optimizerD_signal = optim.Adam(netD_signal.parameters(), lr=lr, betas=(beta1, beta2), eps=1e-08, weight_decay=1e-7)
    optimizerD_freq = optim.Adam(netD_freq.parameters(), lr=lr, betas=(beta1, beta2), eps=1e-08, weight_decay=1e-7)
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2), eps=1e-08, weight_decay=1e-7)

    # torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # optimizerD_signal = torch.optim.SGD(netD_signal.parameters(), lr=lr, momentum=0.9)
    # optimizerD_freq = torch.optim.SGD(netD_freq.parameters(), lr=lr, momentum=0.9)
    # optimizerG = torch.optim.SGD(netG.parameters(), lr=lr, momentum=0.9)

    freq_criterion = nn.MSELoss()
    sig_criterion = nn.MSELoss()

    # freq_criterion = nn.BCELoss()
    # sig_criterion = nn.BCELoss()

    recon_loss = ReconstructLoss()
    # recon_loss = nn.MSELoss()
    # recon_loss = nn.L1Loss()
    print('Done.')

    # -------------------------------------------------------------------------------------------------------------
    ## Start training
    train_loss_history = []
    test_loss_history = []

    best_eval_acc = 0
    print('Start training ...')
    for epoch in range(epoch_num):
        
        d_loss = 0
        g_loss = 0
        total_g_loss = []
        total_d_loss = []
        train_count = 0
        start_time = datetime.now()
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for step, batch in progress_bar:
            PPG = batch['PPG'].unsqueeze(1)
            ECG = batch['ECG'].unsqueeze(1)

            PPG = PPG.to(device)
            ECG = ECG.to(device)
            

            #------------------------------ get freq data --------------------------------------
            ECG_freq = torch.from_numpy(abs(np.fft.fft(ECG.detach().cpu().numpy()))).float()
            ECG_freq = ECG_freq.to(device)
            real_label = torch.ones(ECG.size()[0]).to(device)
            real_label = real_label - 0.1

            PPG_freq = torch.from_numpy(abs(np.fft.fft(PPG.detach().cpu().numpy()))).float()
            PPG_freq = PPG_freq.to(device)
            fake_label = torch.zeros(PPG.size()[0]).to(device)
            # ---------------------------------------------------------------------------------
            
            netD_freq.train()
            netD_signal.train()
            netG.eval()
            #--------------------------- (1) D real update ------------------------------------
            optimizerD_signal.zero_grad()
            optimizerD_freq.zero_grad()
            netD_signal.zero_grad()
            netD_freq.zero_grad()
            total_d_fake_loss = 0
            total_d_real_loss = 0
            Genh_sig = netG(PPG)
            Genh_freq = abs(torch.fft.fft(Genh_sig))
            
            d_sig_real = netD_signal(ECG)
            d_freq_real = netD_freq(ECG_freq)
            
            d_real_sig_loss = sig_criterion(d_sig_real.view(-1), real_label)
            d_real_freq_loss = freq_criterion(d_freq_real.view(-1), real_label)

            d_real_loss = 0.5 * (d_real_freq_loss + d_real_sig_loss)
            # d_real_loss = (d_real_sig_loss)

            d_real_loss.backward()
            total_d_real_loss += d_real_loss
            # ---------------------------------------------------------------------------------
            
            netD_freq.train()
            netD_signal.train()
            netG.eval()
            # --------------------------- (2) D fake update -----------------------------------
            d_sig_fake = netD_signal(Genh_sig.detach())
            d_freq_fake = netD_freq(Genh_freq.detach())

            d_fake_sig_loss = sig_criterion(d_sig_fake.view(-1), fake_label)
            d_fake_freq_loss =  freq_criterion(d_freq_fake.view(-1), fake_label)
            
            d_fake_loss = 0.5 * (d_fake_sig_loss + d_fake_freq_loss)
            # d_fake_loss = d_fake_sig_loss

            d_fake_loss.backward()
            total_d_fake_loss += d_fake_loss
            optimizerD_signal.step()
            optimizerD_freq.step()

            d_loss = d_fake_loss + d_real_loss 
            # --------------------------------------------------------------------------------
            
            netD_freq.eval()
            netD_signal.eval()
            netG.train()
            # ------------------------- (3) G real update ------------------------------------
            optimizerG.zero_grad()
            d_sig_fake = netD_signal(Genh_sig.detach())
            d_freq_fake = netD_freq(Genh_freq.detach())

            d_fake_sig_loss = sig_criterion(d_sig_fake.view(-1), fake_label)
            d_fake_freq_loss =  freq_criterion(d_freq_fake.view(-1), fake_label)
            
            
            g_adv_loss = 0.5 * (d_fake_sig_loss + d_fake_freq_loss)
            # g_adv_loss = d_fake_sig_loss

            g_loss = recon_loss(Genh_sig, ECG) + g_adv_loss

            total_g_loss.append(g_loss.item())
            train_count += 1
            g_loss.backward()
            optimizerG.step()
            # ---------------------------------------------------------------------------------


            cur_time = datetime.now()
            delta_time = cur_time - start_time
            delta_time = timedelta(seconds=delta_time.seconds) 

            progress_bar.set_description(f"Train: [{epoch + 1}/{epoch_num}][{step + 1}/{len(train_dataloader)}] "
                                                        f"GLoss: {np.mean(total_g_loss):.6f} "
                                                        # f"Acc: {np.mean(avg_eval_acc):.3f}%"
                                                        f"Time: {delta_time}\33[0m")
        train_loss_history.append(np.mean(total_g_loss))

        d_val_loss = 0
        g_val_loss = 0
        total_g_val_loss = []
        # val_count = 0
        avg_eval_acc = []
        netD_freq.eval()
        netD_signal.eval()
        netG.eval()
        with torch.no_grad():
            start_time = datetime.now()
            progress_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
            for step, batch in progress_bar:

                PPG = batch['PPG'].unsqueeze(1)
                ECG = batch['ECG'].unsqueeze(1)

                PPG = PPG.to(device)
                ECG = ECG.to(device)
                ECG_freq = torch.from_numpy(abs(np.fft.fft(ECG.detach().cpu().numpy()))).float()
                ECG_freq = ECG_freq.to(device)
                
                # G
                Genh_sig = netG(PPG)
                
                g_val_loss = recon_loss(Genh_sig, ECG)
                
                total_g_val_loss.append(g_val_loss.item())


                cur_time = datetime.now()
                delta_time = cur_time - start_time
                delta_time = timedelta(seconds=delta_time.seconds) 
                progress_bar.set_description(f"Test: [{epoch + 1}/{epoch_num}][{step + 1}/{len(test_dataloader)}] "
                                                            f"GLoss: {np.mean(total_g_val_loss):.6f} "
                                                            # f"Acc: {np.mean(avg_eval_acc):.3f}%"
                                                            f"Time: {delta_time}\33[0m")
            test_loss_history.append(np.mean(total_g_val_loss))
        # save model
        if epoch >= epoch_num - 5:

            torch.save(netG, save_model_path + '/Generator/segan_netG_ep%d_loss_%.3f.pth' %(epoch+1, g_val_loss))
            torch.save(netD_signal, save_model_path + '/Dis_sig/segan_netD_sig_ep%d_loss_%.3f.pth' %(epoch+1, d_val_loss))
            torch.save(netD_freq, save_model_path + '/Dis_freq/segan_netD_freq_ep%d_loss_%.3f.pth' %(epoch+1, d_val_loss))



    # ======================= save AUC ================================
    df_loss = pd.DataFrame({'train_loss': train_loss_history,
                            'val_loss': test_loss_history})
    
    df_loss.to_csv(loss_history_path)
    
    # =================================================================