%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%      This file read signals from the PhysioNet database,    %% 
%%            splits the database into two datasets,           %% 
%%       and saves the datasets as two sepatate .mat files     %%
%%                                                             %%
%%                   Author: Josh Atwal                        %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global nLabels = 4; %Number of output labels
window = 1000; %Length of window breaking signals up

%Directory containing dataset
files = dir('../../reference/dataset/*.mat');

n = length(files); %Number of files in dataset
A = cell(1,n); %Cell matrix to store signals

%Load each signal into cell matrix A
for i=1:n
    filename = ['../../reference/dataset/' files(i).name];
    load(filename)
    A{i} = val;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Read class labels from file 
filename = '../../reference/REFERENCE-v3.csv';
delimiter = ',';
formatSpec = '%*s%s%[^\n\r]';

fileID = fopen(filename,'r'); %Open file

dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'EmptyValue' ,NaN, 'ReturnOnError', false);
fclose(fileID); %Close file

REFERENCEv3 = [dataArray{1:end-1}];

%Clear unwanted variables
clearvars filename delimiter formatSpec fileID dataArray ans val;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parse class labels

%Build string of class labels ('N', 'A', 'O', '~')
LabelStr = '';
[n,~] = size(REFERENCEv3);
for i = 1:n
    LabelStr = [LabelStr REFERENCEv3{i}];
end
   
%Convert string to a numeric vector where:
% 1 = Normal sinus
% 2 = Atrial Fibrillation
% 3 = Other Rhythm
% 4 = Too noisy to classify

Labels = ones(1,n); 
for i = 1:n
    if(LabelStr(i) == 'A')
        Labels(i) = 2;
    elseif (LabelStr(i) == 'O')
        Labels(i) = 3;
    elseif (LabelStr(i) == '~')
        Labels(i) = 4;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Preparing the training set

trainProp = 0.8; %Proportion of signals to go into training set

% Number of each signal type to go into the dataset
nN = round(sum(Labels==1)*trainProp); % Normal signals
nA = round(sum(Labels==2)*trainProp); % AF signals
nO = round(sum(Labels==3)*trainProp); % Other signals
nNo = round(sum(Labels==4)*trainProp); % Noisy signals
ns = [nN nA nO nNo]; 

rng(736729947) %Set random seed

B=A;
BLabels = Labels;

%sum(ns) is the total number of signals to go into training set
train = cell(1,sum(ns)); %Cell matrix of training dataset
trainLabels = zeros(1, sum(ns));

for t=1:nLabels %For each of the four labels
    %For as many of a label are to go into training set
    for j=1:ns(t) 

        %Random index
        r = floor(rand*(sum(BLabels==t)))+1;
        temp = find(BLabels==t);
        %insert signal into training dataset
        train{j} = B{temp(r)}; 

        %Remove signal so as to not re-select it
        B(temp(r)) = [];
        BLabels(temp(r)) = [];
        trainLabels(j) = t;
    end
end

%Randomly shuffle the samples within the dataset 
train=train(randperm(numel(train)));
trainLabels=trainLabels(randperm(numel(trainLabels)));

% Reformat signals into signal segments with function
[sKey, sLabels] = signalSegment(test, testLabels, window)


% Save training set

% train: 6822 length cell array
% S: (64113x1000) matrix used as X in training 
% trainLabels: 6822 length vector of labels
% sKey: 63113 length vector mapping S back to the original dataset
% window: window that breaks up the signal
% sLabels: (64113x4) matrix of labels used as Y in training
% nSubSamples: longest length signal divided by window, used for RNN training
save '../data/trainingSet.mat' train S trainLabels sKey window sLabels


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Testing set

testProp = 1;
nN = round(sum(BLabels==1)*testProp);
nA = round(sum(BLabels==2)*testProp);
nO = round(sum(BLabels==3)*testProp);
nNo = round(sum(Labels==4)*trainProp);
ns = [nN nA nO nNo];

% Reformat signals into signal segments with function
test = B;
testLabels = BLabels;
[sKey, sLabels] = signalSegment(test, testLabels, window)


% Save testing set

% test: 1706 length cell matrix
% S: (16223x1000) matrix used as X in training 
% testLabels: 1706 length vector of labels
% sKey: 16223 length vector mapping S back to the original dataset
% sLabels: (16223x4) matrix of labels used as Y in training
save '../data/testingSet.mat' test S testLabels sKey sLabels


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function for reformatting signals into signal segments

[sKey, sLabels] = function signalSegment(dset, labels, window)

    m = 0; %Total number of signal segments
    n=length(dset);
    for i=1:n
        m = m + floor(length(dset{i})/window);
    end

    % Matrix of signal segments 
    S = zeros(m, window);
    sLabels = ones(1,m);
    k=1;

    % maps signal segments back to original signal indices
    sKey = zeros(1,n); 
    for i = 1:n
        for j = 0: floor(length(dset{i}) / window)-1    
            S(k,:) = dset{i}(((j*window)+1) : (((j+1)*window)));
            sKey(k) = i;
            sLabels(k) = labels(i);
            k = k + 1;
        end
    end

    %One-hot encoding
    sLabs = zeros(length(sLabels),nLabels);
    for i=1:length(sLabels)
        sLabs(i,sLabels(i)) = 1;
    end
    sLabels = sLabs;

end