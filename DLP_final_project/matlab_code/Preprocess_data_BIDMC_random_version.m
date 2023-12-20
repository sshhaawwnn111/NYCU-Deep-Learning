clc;clear;
close all;

datadir = "../data/BIDMC/*signal*.csv";
savedir = "../data/BIDMC_preprocess_30s_ripple_filt_ECG12_v2";
% savedir = "../data/BIDMC_preprocess_30s_check_plot_img_ripple_filt";
if ~exist(savedir, 'dir')
    mkdir(savedir)
else
    disp("dir exist");
end



fps = 125;
slice_sec = 30; % 30 sec
step = 30; % 1s
save_fig = 0;

slice_sec_fram = fps * slice_sec;

file_dir = dir(datadir);

[A_PPG,B_PPG,C_PPG,D_PPG] = cheby2(4, 20, [0.5 10]./(fps/2), 'bandpass');
[filter_SOS_PPG, g_PPG] = ss2sos(A_PPG,B_PPG,C_PPG,D_PPG);

[A_ECG,B_ECG,C_ECG,D_ECG] = cheby2(4, 20, [0.5 10]./(fps/2), 'bandpass');
[filter_SOS_ECG, g_ECG] = ss2sos(A_ECG,B_ECG,C_ECG,D_ECG);

for i=1:length(file_dir)

    disp(['Processing: ' file_dir(i).name])
    file_path = [file_dir(i).folder '/' file_dir(i).name];
    sig_data = readtable(file_path);
    
    all_PPG = sig_data.PLETH;
    all_ECG = sig_data.II;
    all_PPG = preprocess_signal(all_PPG, filter_SOS_PPG, g_PPG, 0);
    all_ECG = preprocess_signal(all_ECG, filter_SOS_ECG, g_ECG, 0);
    

    [c,lags] = xcorr(all_PPG, all_ECG, 125);
    [max_c, max_c_idx] = max(c);
    all_PPG = circshift(all_PPG, lags(max_c_idx));

    savedir_subject_train_dir = strcat(savedir, '/train/subject', num2str(i));
    savedir_subject_train_fig_dir = strcat(savedir, '/train/pictue/subject', num2str(i));
    savedir_subject_test_dir = strcat(savedir, '/test/subject', num2str(i));
    savedir_subject_test_fig_dir = strcat(savedir, '/test/pictue/subject', num2str(i));

    
    if ~exist(savedir_subject_train_dir, 'dir')
        mkdir(savedir_subject_train_dir)
        if save_fig
            mkdir(savedir_subject_train_fig_dir)
        end
    else
        disp("dir exist");
    end

    if ~exist(savedir_subject_test_dir, 'dir')
        mkdir(savedir_subject_test_dir)
        if save_fig
            mkdir(savedir_subject_test_fig_dir)
        end
    else
        disp("dir exist");
    end

    for j=1:floor(((length(sig_data.PLETH)-slice_sec_fram)/(step*fps)) + 1)
        start_point = (j-1) * (step*fps) + 1;
        end_point = start_point + slice_sec_fram-1;

        PPG = all_PPG(start_point:end_point);
        ECG = all_ECG(start_point:end_point);
        [~, ripple_SQI] = getRippleSQI(PPG, 23, 14, fps);
        [~, ripple_SQI_2] = getRippleSQI(ECG, 23, 14, fps);
        

        if ripple_SQI>15
            

            PPG = preprocess_signal(PPG, filter_SOS_PPG, g_PPG, 1);
            ECG = preprocess_signal(ECG, filter_SOS_ECG, g_ECG, 1);
            
            if j < floor(((length(sig_data.PLETH)-slice_sec_fram)/(step*fps)) + 1) * 0.8
                % Train

                if save_fig
                    savedir_fig_subject_seg_path = strcat(savedir_subject_train_fig_dir, '/subject', num2str(i), '_seg_', num2str(j), '.jpg');
                    a = figure;
                    plot(PPG);hold on
                    plot(ECG);
                    saveas(a, savedir_fig_subject_seg_path)
                    close(a);
                end
                
                savedir_subject_seg_path = strcat(savedir_subject_train_dir, '/subject', num2str(i), '_seg_', num2str(j), '.csv');
            else
                % Test
                if save_fig
                    savedir_fig_subject_seg_path = strcat(savedir_subject_test_fig_dir, '/subject', num2str(i), '_seg_', num2str(j), '.jpg');
                    a = figure;
                    plot(PPG);hold on
                    plot(ECG);
                    saveas(a, savedir_fig_subject_seg_path)
                    close(a);
                end
                
                savedir_subject_seg_path = strcat(savedir_subject_test_dir, '/subject', num2str(i), '_seg_', num2str(j), '.csv');
            end
    
            T = table(PPG, ECG);
            writetable(T, savedir_subject_seg_path)
            clear T;
        end

    end
end


function filted_signal = preprocess_signal(signal, filter, g, normalize_flag)

    filted_signal = filtfilt(filter, g, signal);
    if normalize_flag
        filted_signal = (filted_signal-mean(filted_signal))/std(filted_signal);
        filted_signal = (filted_signal - min(filted_signal))/(max(filted_signal)-min(filted_signal)) * 2 -1;

    end
end


function [ret, ripple_SQI] = getRippleSQI(POS, high_cutof_freq, low_cutof_freq, fps)

    % [In] r_buf, g_buf, b_buf  Input original rgb three channels signal.
    % [In] POS_window_length    Window length for calculating POS signal.    
    % [In] fps                  Frame per second for filter.
    % [out] ret                 success or not.
    % [out] Ripple SQI          std(POS_lp_24) / std(POS_lp_24 - POS_lp_14);
    
    % 此SQI反映波型擾動程度，SQI越低擾動越嚴重。

%     high_cutof_freq = 23;
%     low_cutof_freq  = 14;
    
    ret = false;

    Fn = fps/2;
    [A,B,C,D] = cheby2(4, 20, [0.5 high_cutof_freq] ./ Fn, 'bandpass');
    [filter_SOS_high, g_high] = ss2sos(A,B,C,D);
    
    [A,B,C,D] = cheby2(4, 20, [0.5 low_cutof_freq] ./ Fn, 'bandpass');
    [filter_SOS_low, g_low] = ss2sos(A,B,C,D);   
    
    POS_lp_high = filtfilt(filter_SOS_high, g_high, POS);
    POS_lp_low  = filtfilt(filter_SOS_low, g_low, POS);
    
    if POS_lp_high ~= POS_lp_low
        
        ret = true;
        
        ripple_SQI = std(POS_lp_low) / std(POS_lp_high - POS_lp_low);
    else
        disp('POS_lp_high is equal to POS_lp_low');
    end
        
end

