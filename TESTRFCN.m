% Load the trained network and test dataset
load('rfcn_trainedNetwork.mat', 'net', 'imdsTest');

% Prompt user to select an image file
[file, path] = uigetfile({'*.jpg;*.png;*.bmp', 'Image Files (*.jpg, *.png, *.bmp)'}, 'Select an Image for Classification');
if isequal(file, 0)
    disp('No file selected. Exiting...');
    return;
end

% Read and preprocess the image
imagePath = fullfile(path, file);
inputImage = imread(imagePath);
resizedImage = imresize(inputImage, [128 128]); % Resize to match input size

% Classify the image
[predictedLabel, scores] = classify(net, resizedImage);

% Display the result
figure;
imshow(inputImage);
title(sprintf('Predicted Label: %s', string(predictedLabel)));

% Calculate metrics for the predicted class
actualLabels = imdsTest.Labels;
classNames = categories(actualLabels);

% Find the index of the predicted label
predictedIndex = find(classNames == predictedLabel);

% Display metrics for the predicted class
confMat = confusionmat(actualLabels, classify(net, augmentedImageDatastore([128 128 3], imdsTest)));
TP = confMat(predictedIndex, predictedIndex); % True Positives
FP = sum(confMat(:, predictedIndex)) - TP;   % False Positives
FN = sum(confMat(predictedIndex, :)) - TP;   % False Negatives 
%TN = sum(confMat(:)) - (TP + FP + FN);   % True Negative


%accuracy = (TP + TN) / sum(confMat(:));   % Calculate Accuracy
precision = TP / (TP + FP + eps);
recall = TP / (TP + FN + eps);
f1Score = 2 * (precision * recall) / (precision + recall + eps);

% Display scores and metrics
fprintf('\nClassification Scores for Each Class:\n');
for i = 1:numel(classNames)
    fprintf('%s: %.2f%%\n', classNames{i}, scores(i) * 100);
end

fprintf('\nPerformance Metrics for Predicted Class (%s):\n', string(predictedLabel));
fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('F1-Score: %.2f\n', f1Score);
