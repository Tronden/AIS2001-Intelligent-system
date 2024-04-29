%% Task 2.1: Implementing a Neuro-Fuzzy Inference System

% Step 1: Create Random Input Data
% Generating random input data for three input features.
numData = 1000; % Number of data points
input1 = rand(numData, 1) * (maxInput1 - minInput1) + minInput1; % Scale to input range
input2 = rand(numData, 1) * (maxInput2 - minInput2) + minInput2;
input3 = rand(numData, 1) * (maxInput3 - minInput3) + minInput3;

inputs = [input1, input2, input3]; % Combine into a single matrix

% Step 2: Generate the Targets
% Use the fuzzy inference system to generate output for the given inputs.
outputs = evalfis([input1, input2, input3], fis);

% Step 3: Divide Data into Train and Test Sets
% Randomly partition the dataset into training and testing sets.
trainRatio = 0.7;
trainIdx = randperm(numData, round(numData * trainRatio));
testIdx = setdiff(1:numData, trainIdx);

trainInputs = inputs(trainIdx, :);
trainOutputs = outputs(trainIdx);
testInputs = inputs(testIdx, :);
testOutputs = outputs(testIdx);

% Step 4: Create and Train the Neuro-Fuzzy Inference System
% Generating an initial FIS structure from data
genOpt = genfisOptions('GridPartition');
genOpt.NumMembershipFunctions = 3; % Number of membership functions
genOpt.InputMembershipFunctionType = "gaussmf"; % Type of membership function
initialFis = genfis(trainInputs, trainOutputs, genOpt);

% Setting training options
anfisOpt = anfisOptions;
anfisOpt.EpochNumber = 100; % Number of training epochs

% Training the ANFIS model
[trainedFis, trainError] = anfis([trainInputs, trainOutputs], initialFis, anfisOpt);

% Step 5: Evaluate the Model on Test Data
testFisOutput = evalfis(testInputs, trainedFis);

% Step 6: Compute Performance Metrics
mse = immse(testFisOutput, testOutputs); % Mean Squared Error
rmse = sqrt(mse); % Root Mean Squared Error
meanError = mean(testFisOutput - testOutputs);
stdDeviation = std(testFisOutput - testOutputs);

% Step 7: Plot Outputs
figure;
plot(testOutputs, 'bo-', 'DisplayName', 'Actual Outputs');
hold on;
plot(testFisOutput, 'r*-', 'DisplayName', 'ANFIS Outputs');
title('Comparison of Test Outputs and ANFIS Outputs');
xlabel('Sample Number');
ylabel('Output');
legend show;
grid on;
hold off;

% Step 8: Beyond the Universe of Discourse
% Generating inputs outside the specified range and comparing outputs
outOfRangeInput = [maxInput1 * 1.2, maxInput2 * 1.2, maxInput3 * 1.2; % Example out-of-range input
                   minInput1 * 1.2, minInput2 * 1.2, minInput3 * 1.2];
outOfRangeOutput = evalfis(outOfRangeInput, fis);
outOfRangeANFISOutput = evalfis(outOfRangeInput, trainedFis);

% Displaying outputs beyond the specified range
fprintf('Outputs for input beyond the range using FIS and ANFIS:\n');
disp([outOfRangeOutput, outOfRangeANFISOutput]);