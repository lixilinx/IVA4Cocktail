clear all; close all; clc
fs = 16000; % sampling rate
wav_lens = [fs, 2*fs, 4*fs, 8*fs, 16*fs, 32*fs, 64*fs];
N = 3;
SIRs = zeros(5, length(wav_lens));
num_iter = 50;
for iter = 1 : num_iter
    for len = 1 : length(wav_lens)
        [mxts, mxts_gt] = generate_mixtures( N, wav_lens(len), fs );
        %audiowrite('inputs.wav', mxts'/max(abs(mxts(:))), fs);
        %% test IVA with different source priors
        for test_case = 0 : 4
            [y, y_gt] = iva_batch( mxts, test_case, mxts_gt ); % need to halve the learning rate in iva_batch for student's t source prior
            
            energy_gt = squeeze(sum(y_gt.^2, 3));
            all_perms = perms([1:N]); % find the best permutation
            highest_energy = 0;
            for i = 1 : size(all_perms, 1)
                energy = 0;
                for j = 1 : N
                    energy = energy + energy_gt(j, all_perms(i,j));
                end
                highest_energy = max(highest_energy, energy);
            end
            SIR = 1/(sum(sum(energy_gt))/highest_energy - 1);
            SIRs(test_case+1, len) = SIRs(test_case+1, len) + SIR;
            %audiowrite(['outputs',num2str(test_case), '.wav'], y'/max(abs(y(:))), fs);
        end
    end
    10*log10( SIRs/iter )
end
semilogx(wav_lens/fs, 10*log10(SIRs(1,:)/num_iter), '.r-');
hold on; semilogx(wav_lens/fs, 10*log10(SIRs(2,:)/num_iter), 'm-x');
hold on; semilogx(wav_lens/fs, 10*log10(SIRs(3,:)/num_iter), 'c-+');
hold on; semilogx(wav_lens/fs, 10*log10(SIRs(4,:)/num_iter), 'g-o');
hold on; semilogx(wav_lens/fs, 10*log10(SIRs(5,:)/num_iter), 'b-*');
xlabel('Length of speech (s)'); ylabel('SIR (dB)');
legend('Laplace', 'GGD', 'Student', 'Non-Spherical', 'FNN');
