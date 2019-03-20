%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function for preparing a dataset of spectrograms 
%
% Author: Josh Atwal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Training spectrogram set
load ../data/trainingSet_w1000.mat
Spec = genSpec(S);
save ../data/trainingSpectrogram.mat Spec sKey sLabels trainLabels window nSubSamples

clear

%Testing spectrogram set
load ../data/validationSet_w1000.mat
Spec = genSpec(S);
save ../data/validationSpectrogram.mat Spec sKey sLabels testLabels 


%Function for generating matrix of spectrograms
function [Spec] =  genSpec(S)
    fs=300;
    t=8;
    f=50;
    dsf = 15; %downsample factor
    Spec = zeros(length(S),t,f,1);
    figure
    for i=1:length(S)
        %Plot spectrogram, and save figure image data
        spectrogram(S(i,:), [],[],[], fs,'yaxis');
        data = get(get(gcf,'Children'),'Children');
        data = get(data{2}, 'CData');
        s = data(1:f,1:t);
        Spec(i,1:f,1:t,1) = (s - mean(s(:)))/std(s(:));
    end
end
