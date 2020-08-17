function list = list_all_wavs( dir0 )
% list all wav files in a dir
list = dir([dir0, '**']);
filter = zeros(size(list));
for j = 1 : length(list)
    filter(j) = (~list(j).isdir) && (endsWith(list(j).name, '.wav') || endsWith(list(j).name, '.WAV'));
end
list = list(filter==1);