% Part1 Generating random inputs
minDelay = 0;
maxDelay = 0.7;
minServers = 0;
maxServers = 1;
minUtilization = 0;
maxUtilization = 1;
numSamples = 1000;
mean_delay = rand(numSamples, 1) * (maxDelay - minDelay) + minDelay;  
number_of_servers = randi([minServers, maxServers], numSamples, 1);  
repair_utilization = rand(numSamples, 1) * (maxUtilization - minUtilization) + minUtilization;  

data = [mean_delay, number_of_servers, repair_utilization];

% Part2 generate target using fis
fis = readfis('MamdaniType1.fis');
outputs = evalfis(fis, data);
data = [data, outputs];

% Part3 split the data
numObservations = size(data, 1);
shuffledIndices = randperm(numObservations);
numTrain = floor(0.8 * numObservations);  % 80% for training

% Training set
trainIndices = shuffledIndices(1:numTrain);
X_train = data(trainIndices, 1:end-1);
y_train = data(trainIndices, end);

% Test set
testIndices = shuffledIndices(numTrain+1:end);
X_test = data(testIndices, 1:end-1);
y_test = data(testIndices, end);

% Part4 Create and train neuro-fuzzy system
num_epochs = 100; 
options = anfisOptions('InitialFIS', 3, 'EpochNumber', num_epochs);
anfis_model = anfis([X_train, y_train], options);

% Part5 Evaluate data
y_pred = evalfis(anfis_model, X_test);
 

% Part6 Calcutae errors MSE, RMSE
mse = immse(y_test, y_pred); 
rmse = sqrt(mse); 
errors = y_pred - y_test;
mean_errors = mean(errors);
std_errors = std(errors);
fprintf('Mean of errors: %f\n', mean_errors);
fprintf('Standard deviation of errors: %f\n', std_errors);

% Part7 plotting results
figure;
scatter(y_test, y_pred, 'filled');
hold on;
p = polyfit(y_test, y_pred, 1); 
yfit = polyval(p, y_test);
plot(y_test, yfit, 'r-');
legend('Data Points', 'Fit Line');
xlabel('Actual Spares');
ylabel('Predicted Spares');
title('Scatter Plot with Linear Fit');
grid on;
hold off;


% Part 8 Generate out-of-range input data
out_range_samples = 10;  
out_range_data = [rand(out_range_samples, 1) * (1.2 - maxDelay) + maxDelay, ...  % Mean delay
                  randi([maxServers + 1, maxServers + 5], out_range_samples, 1), ...  % Number of servers
                  rand(out_range_samples, 1) * (1.2 - maxUtilization) + maxUtilization];  % Repair utilization

out_range_outputs_fis = evalfis(fis, out_range_data);
out_range_outputs_anfis = evalfis(anfis_model, out_range_data);
disp('FIS Outputs for Out-of-Range Data:');
disp(out_range_outputs_fis);
disp('ANFIS Outputs for Out-of-Range Data:');
disp(out_range_outputs_anfis);