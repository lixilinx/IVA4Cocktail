function [mxts, mxts_gt] = generate_mixtures( N, wav_len, fs )
% generate N mixtures with wav_len samples and sampling rate fs
%
% outputs:
% mxts: generated mixtures
% mxts_gt: ground truth mixtures. only for performance reporting purpose
persistent wav_list;
if isempty(wav_list)
    wav_list = list_all_wavs( 'E:\Xi-Lin Li\Old_2011_2015\BinWiseVAD\database\timit\' ); % use timit as test set. 
end

L = 5; W = 4; H = 3; % [Length, Width, Height] of room

%% sampling microphone locations
mic_ctr = [2, 2, 1.5];
mic_locs = []; % random microphone locations; |loc - mic_ctr| < 0.1 meter
while size(mic_locs, 1) < N
    loc = mic_ctr + 0.2*rand(1, 3) - 0.1;
    if norm(loc - mic_ctr) < 0.1
        mic_locs = [mic_locs; loc];
    end
end

%% sampling source locations
src_locs = []; % random source locations; |loc - mic_ctr| > 1 meter
while size(src_locs, 1) < N
    loc = [L, W, H].*rand(1, 3);
    if norm(loc - mic_ctr) > 1
        src_locs = [src_locs; loc];
    end
end

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

%% generate mixtures using image source method
mxts = zeros(N, wav_len); % mixtures
mxts_gt = zeros(N, N, wav_len); % ground truth mixtures. only for performance reporting purpose
for m = 1 : N % mic index
    for n = 1 : N % source index
        % [h]=rir(fs, mic, n, r, rm, src)
        h = rir(3*fs, mic_locs(m,:), 5, 0.25, [L,W,H], src_locs(n,:));
        h = decimate(h, 3); h = h(1:round(0.1*fs)); % get fractionally delay in this way. Truncated to 100 ms length
        mxt = conv(h', srcs(n,:));
        mxt = mxt(1 : wav_len);
        mxts(m,:) = mxts(m,:) + mxt;
        mxts_gt(m,n,:) = reshape(mxt, 1, 1, wav_len);
    end
end