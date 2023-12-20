clc;clear;
close all;

datadir = "../data/BIDMC/*signal*.csv";
savedir = "../data/BIDMC_preprocess_20s";
if ~exist(savedir, 'dir')
    mkdir(savedir)
else
    disp("dir exist");
end



fps = 125;
slice_sec = 20; % 30 sec
slice_sec_fram = fps * slice_sec;

file_dir = dir(datadir);

[A_PPG,B_PPG,C_PPG,D_PPG] = cheby2(4, 20, [0.5 10]./(fps/2), 'bandpass');
[filter_SOS_PPG, g_PPG] = ss2sos(A_PPG,B_PPG,C_PPG,D_PPG);

[A_ECG,B_ECG,C_ECG,D_ECG] = cheby2(4, 20, [0.5 15]./(fps/2), 'bandpass');
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

    if i < 41
        savedir_subject_dir = strcat(savedir, '/train/subject', num2str(i));
    else
        savedir_subject_dir = strcat(savedir, '/test/subject', num2str(i));
    end
    
    if ~exist(savedir_subject_dir, 'dir')
        mkdir(savedir_subject_dir)
    else
        disp("dir exist");
    end

    for j=1:floor(length(sig_data.PLETH)/slice_sec_fram)-1
        start_point = j * slice_sec_fram;
        end_point = (j+1) * slice_sec_fram -1;

        PPG = all_PPG(start_point:end_point);
        ECG = all_ECG(start_point:end_point);

        PPG = preprocess_signal(PPG, filter_SOS_PPG, g_PPG, 1);
        ECG = preprocess_signal(ECG, filter_SOS_ECG, g_ECG, 1);

        savedir_subject_seg_path = strcat(savedir_subject_dir, '/subject', num2str(i), '_seg_', num2str(j), '.csv');

        T = table(PPG, ECG);
        writetable(T, savedir_subject_seg_path)
        clear T;
    end
end


function filted_signal = preprocess_signal(signal, filter, g, normalize_flag)

    filted_signal = filtfilt(filter, g, signal);
    if normalize_flag
%         filted_signal = (filted_signal-mean(filted_signal))/std(filted_signal);
        filted_signal = (filted_signal - min(filted_signal))/(max(filted_signal)-min(filted_signal));

    end
end


