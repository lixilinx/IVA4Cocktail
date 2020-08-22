clear all; close all; clc
fs = 16000; % sampling rate
wav_list = list_all_wavs( 'E:\Xi-Lin Li\Old_2011_2015\BinWiseVAD\database\timit\' ); % use timit as test set
wav_len = 30*fs;
N = 2; lr = 0.03;
num_iter = 10;
SIRs = zeros(wav_len, 3);
for iter = 1 : num_iter
    %% read random wave file as source
    srcs = zeros(N, wav_len);
    for n = 1 : N
        src = [];
        while length(src) < wav_len
            i = ceil(rand*length(wav_list));
            s = audioread([wav_list(i).folder, '\', wav_list(i).name])';
            src = [src, s(513:end)]; % timit wave is NIST sphere file, not real wav format. May need to discard the first 512 samples
        end
        srcs(n,:) = src(1:wav_len);
    end
    mxts_gt = zeros(N, N, wav_len);
    mxts_gt(1,1,:) = reshape(filter([1,2,1],[1,0,0.17], srcs(1,:)), 1,1,wav_len);
    mxts_gt(1,2,:) = reshape(filter([1,-2,1],[1,0,0.17], srcs(2,:)), 1,1,wav_len);
    mxts_gt(2,1,:) = reshape(filter([1,-2,1],[1,0,0.17], srcs(1,:)), 1,1,wav_len);
    mxts_gt(2,2,:) = reshape(filter([1,2,1],[1,0,0.17], srcs(2,:)), 1,1,wav_len);
    mxts = zeros(N, wav_len);
    mxts(1,:) = squeeze(mxts_gt(1,1,:) + mxts_gt(1,2,:));
    mxts(2,:) = squeeze(mxts_gt(2,1,:) + mxts_gt(2,2,:));
    
    %% test IVA with Lap, FNN and RNN models
    for test_case = 0 : 2
        [y, y_gt] = iva_online( mxts, test_case, mxts_gt, lr );
        for m=1:N
            for n=1:N
                y_gt(m,n,:) = reshape(filter([1,-1],1,squeeze(y_gt(m,n,:))),1,1,wav_len);
            end
        end
        energy12 = squeeze(y_gt(1, 1, :)).^2 + squeeze(y_gt(2, 2, :)).^2 + eps;
        energy21 = squeeze(y_gt(1, 2, :)).^2 + squeeze(y_gt(2, 1, :)).^2 + eps;
        SIR = filter(1, [1, 1/fs - 1], energy12) ./ filter(1, [1, 1/fs - 1], energy21);
        if sum(energy12) < sum(energy21)
            SIR = 1./SIR;
        end
        SIRs(:, test_case+1) = SIRs(:, test_case+1) + SIR;
    end
end
plot([1:wav_len]/fs, 10*log10(SIRs(:,1)/num_iter), 'r');
hold on; plot([1:wav_len]/fs, 10*log10(SIRs(:,2)/num_iter), 'b');
hold on; plot([1:wav_len]/fs, 10*log10(SIRs(:,3)/num_iter), 'k');
xlabel('Time (s)'); ylabel('SIR (dB)');
legend('Multivariate Lap', 'Estimated, FNN', 'Estimated, RNN');