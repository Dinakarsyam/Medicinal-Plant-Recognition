%% Medicinal Plant Classification Using Deep Learning with Region-Based Fully Convolutional Networks
% Clear workspace
% NRI COLLEGE
clc;
clear;
close all;
msgbox('WELCOME TO MEDICINAL PLANTS CLASSIFICATION');


% Load Image Data
dataFolder = 'DATA - Copy'; % Replace with the path to your dataset
imds = imageDatastore(dataFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Display the count of each label
disp('Count of each label:');
countEachLabel(imds);

% Split data into training and testing sets
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

% Resize and augment images
inputSize = [128 128 3]; % Resized image size
augmenter = imageDataAugmenter( ...
    'RandRotation', [-20, 20], ...
    'RandXTranslation', [-15, 15], ...
    'RandYTranslation', [-15, 15], ...
    'RandScale', [0.8, 1.2], ...
    'RandXReflection', true);
augmentedTrain = augmentedImageDatastore(inputSize, imdsTrain, 'DataAugmentation', augmenter);
augmentedTest = augmentedImageDatastore(inputSize, imdsTest);

% Define the R-FCN architecture
layers = [
    % Input layer
    imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'zscore')
    
    % Convolutional Block 1
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'batchnorm1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')
    
    % Convolutional Block 2
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'batchnorm2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')
    
    % Convolutional Block 3
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'batchnorm3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool3')
    
    % Convolutional Block 4 (Additional Block for R-FCN)
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv4')
    batchNormalizationLayer('Name', 'batchnorm4')
    reluLayer('Name', 'relu4')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool4')
    
    % Region Proposal Network (RPN)
    convolution2dLayer(1, 512, 'Name', 'rpn_conv1') % Generate region proposals
    reluLayer('Name', 'rpn_relu1')
    
    % Position-Sensitive Score Maps (for R-FCN)
    convolution2dLayer(1, 512, 'Name', 'score_maps')
    reluLayer('Name', 'relu_score')
    
    % Flatten Layer
    flattenLayer('Name', 'flatten')
    
    % Fully Connected Layers for Classification
    fullyConnectedLayer(512, 'Name', 'fc1')
    reluLayer('Name', 'relu_fc1')
    dropoutLayer(0.5, 'Name', 'dropout1')
    fullyConnectedLayer(7, 'Name', 'fc2') % Number of classes
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% Specify Training Options
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augmentedTest, ...
    'ValidationFrequency', 50, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto'); % Use GPU if available

% Train the Network
net = trainNetwork(augmentedTrain, layers, options);
analyzeNetwork(net);
% Evaluate the Network
predictedLabels = classify(net, augmentedTest);
actualLabels = imdsTest.Labels;

% Calculate accuracy
accuracy = sum(predictedLabels == actualLabels) / numel(actualLabels);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

% Display confusion matrix
confMat = confusionmat(actualLabels, predictedLabels);
confusionchart(actualLabels, predictedLabels);
title('Confusion Matrix');

% Calculate precision, recall, and F1-score
numClasses = numel(categories(actualLabels));
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
f1Score = zeros(numClasses, 1);

for i = 1:numClasses
    TP = confMat(i, i); % True Positives
    FP = sum(confMat(:, i)) - TP; % False Positives
    FN = sum(confMat(i, :)) - TP; % False Negatives
    
    % Avoid division by zero
    precision(i) = TP / (TP + FP + eps);
    recall(i) = TP / (TP + FN + eps);
    f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i) + eps);
end

% Display results
classNames = categories(actualLabels);
fprintf('\nPerformance Metrics:\n');
fprintf('Class\t\tPrecision\tRecall\t\tF1-Score\n');
for i = 1:numClasses
    fprintf('%s\t\t%.2f\t\t%.2f\t\t%.2f\n', classNames{i}, precision(i), recall(i), f1Score(i));
end

% Save the trained network
save('rfcn_trainedNetwork.mat', 'net');
save('rfcn_trainedNetwork.mat', 'net', 'imdsTest');

