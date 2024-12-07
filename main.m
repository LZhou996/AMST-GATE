doTraining = true;
%% Load data
data = load('D:\Dataset\CMAPSS\train_FD001.txt');
ID = data(:,1);
featuresTrain = [data(:,7:9),data(:,12:14),data(:,16:20),data(:,22),data(:,25:26)];
[featuresTrain,muData,sigmaData] = normalize(featuresTrain);%Normalized data
windowSize = 31;% Specify window length
[XTrain,ATrain,TTrain] = processTrainData(featuresTrain,ID,windowSize);
dsXTrain = arrayDatastore(XTrain,IterationDimension=3);
dsATrain = arrayDatastore(ATrain,IterationDimension=3);
dsTTrain = arrayDatastore(TTrain,IterationDimension=2);
dsTrain = combine(dsXTrain,dsATrain,dsTTrain);% Training data

%% Define model
% Initialize AMS-GAT parameters
parameters = struct;
embeddingDimension = 96;
numHiddenUnits = 1024;
numChannels = size(featuresTrain,2);
inputSize = numChannels+1;
sz = [embeddingDimension inputSize];

% 1st AMS-GAT
% Self-attention mechanism
sz = [embeddingDimension windowSize];
numOut = embeddingDimension;
numIn = windowSize;
parameters.graphattn1A.weights.linear = initializeGlorot(sz,numOut,numIn);
parameters.graphattn1A.weights.attention = initializeGlorot([2 numOut],1,2*numOut);

% ExpSparse self-attention mechanism
sz = [embeddingDimension windowSize];
numOut = embeddingDimension;
numIn = windowSize;
parameters.graphattn1B.weights.linear = initializeGlorot(sz,numOut,numIn);
parameters.graphattn1B.weights.attention = initializeGlorot([2 numOut],1,2*numOut);

% ExpSparse self-attention mechanism
sz = [embeddingDimension windowSize];
numOut = embeddingDimension;
numIn = windowSize;
parameters.graphattn1C.weights.linear = initializeGlorot(sz,numOut,numIn);
parameters.graphattn1C.weights.attention = initializeGlorot([2 numOut],1,2*numOut);

% GSAM
filterSize = [3 1];
Channels = 1;
numFilters = 1;
sz = [filterSize Channels numFilters];
numOut = prod(filterSize) * numFilters;
numIn = prod(filterSize) * numFilters;
parameters.conv1A.Weights = initializeGlorot(sz,numOut,numIn);
parameters.conv1A.Bias = initializeZeros([numFilters 1]);

filterSize = [3 1];
Channels = 1;
numFilters = 1;
sz = [filterSize Channels numFilters];
numOut = prod(filterSize) * numFilters;
numIn = prod(filterSize) * numFilters;
parameters.conv1B.Weights = initializeGlorot(sz,numOut,numIn);
parameters.conv1B.Bias = initializeZeros([numFilters 1]);

% 2nd AMS-GAT
% Self-attention mechanism
sz = [embeddingDimension embeddingDimension];
numOut = embeddingDimension;
numIn = embeddingDimension;
parameters.graphattn2A.weights.linear = initializeGlorot(sz,numOut,numIn);
parameters.graphattn2A.weights.attention = initializeGlorot([2 numOut],1,2*numOut);

% ExpSparse self-attention mechanism
sz = [embeddingDimension embeddingDimension];
numOut = embeddingDimension;
numIn = embeddingDimension;
parameters.graphattn2B.weights.linear = initializeGlorot(sz,numOut,numIn);
parameters.graphattn2B.weights.attention = initializeGlorot([2 numOut],1,2*numOut);

% ExpSparse self-attention mechanism
sz = [embeddingDimension embeddingDimension];
numOut = embeddingDimension;
numIn = embeddingDimension;
parameters.graphattn2C.weights.linear = initializeGlorot(sz,numOut,numIn);
parameters.graphattn2C.weights.attention = initializeGlorot([2 numOut],1,2*numOut);

% GSAM
filterSize = [3 1];
Channels = 1;
numFilters = 1;
sz = [filterSize Channels numFilters];
numOut = prod(filterSize) * numFilters;
numIn = prod(filterSize) * numFilters;
parameters.conv2A.Weights = initializeGlorot(sz,numOut,numIn);
parameters.conv2A.Bias = initializeZeros([numFilters 1]);

filterSize = [3 1];
Channels = 1;
numFilters = 1;
sz = [filterSize Channels numFilters];
numOut = prod(filterSize) * numFilters;
numIn = prod(filterSize) * numFilters;
parameters.conv2B.Weights = initializeGlorot(sz,numOut,numIn);
parameters.conv2B.Bias = initializeZeros([numFilters 1]);

% Initialize ABiTCN parameters
% Positive residual block
filterSize = [14 3];
Channels = 1;
numFilters = 16;
sz = [filterSize Channels numFilters];
numOut = prod(filterSize) * numFilters;
numIn = prod(filterSize) * numFilters;
parameters.conv11.Weights = initializeGlorot(sz,numOut,numIn);
parameters.conv11.Bias = initializeZeros([numFilters 1]);

filterSize = [14 3];
Channels = 16;
numFilters = 16;
sz = [filterSize Channels numFilters];
numOut = prod(filterSize) * numFilters;
numIn = prod(filterSize) * numFilters;
parameters.conv12.Weights = initializeGlorot(sz,numOut,numIn);
parameters.conv12.Bias = initializeZeros([numFilters 1]);

filterSize = [1 1];
Channels = 1;
numFilters = 16;
sz = [filterSize Channels numFilters];
numOut = prod(filterSize) * numFilters;
numIn = prod(filterSize) * numFilters;
parameters.conv13.Weights = initializeGlorot(sz,numOut,numIn);
parameters.conv13.Bias = initializeZeros([numFilters 1]);

% Reverse residual block
filterSize = [14 3];
Channels = 1;
numFilters = 16;
sz = [filterSize Channels numFilters];
numOut = prod(filterSize) * numFilters;
numIn = prod(filterSize) * numFilters;
parameters.conv21.Weights = initializeGlorot(sz,numOut,numIn);
parameters.conv21.Bias = initializeZeros([numFilters 1]);

filterSize = [14 3];
Channels = 16;
numFilters = 16;
sz = [filterSize Channels numFilters];
numOut = prod(filterSize) * numFilters;
numIn = prod(filterSize) * numFilters;
parameters.conv22.Weights = initializeGlorot(sz,numOut,numIn);
parameters.conv22.Bias = initializeZeros([numFilters 1]);

filterSize = [1 1];
Channels = 1;
numFilters = 16;
sz = [filterSize Channels numFilters];
numOut = prod(filterSize) * numFilters;
numIn = prod(filterSize) * numFilters;
parameters.conv23.Weights = initializeGlorot(sz,numOut,numIn);
parameters.conv23.Bias = initializeZeros([numFilters 1]);

% Bidirectional temporal feature fusion
parameters.alpha = dlarray(rand(1,1));

% Initialize fully connected layer parameters
numResponses = 1;
sz = [numResponses embeddingDimension*numChannels*numFilters];
numOut = numResponses;
numIn = embeddingDimension*numChannels*numFilters;
parameters.fc.weights = initializeGlorot(sz,numOut,numIn);
parameters.fc.bias = initializeZeros([numOut,1]);

%% Training model
% Specify training options
numEpochs = 10;
miniBatchSize = 100;
learnRate = 0.001;
mbq = minibatchqueue(dsTrain,...
    MiniBatchSize=miniBatchSize,...
    OutputAsDlarray=[1 1 0],...
    OutputEnvironment = ["cpu" "cpu" "cpu"]);
trailingAvg = [];
trailingAvgSq = [];% Initialize Adam optimizer
L=[];

% Start training
if doTraining
    figure(1)
    C = colororder;
    lineLossTrain = animatedline(Color=C(2,:));
    ylim([0 inf]);
    xlabel('Iteration');
    ylabel('Loss');
    grid on

    iteration = 0;
    start = tic;
    % Loop over epochs
    for epoch = 1:numEpochs
        % Shuffle data
        shuffle(mbq)
        % Loop over mini-batches
        while hasdata(mbq)
            iteration = iteration+1;
            % Read mini-batches of data
            [X,A,T] = next(mbq); T = T';  
            % Evaluate the model loss and gradients using dlfeval and the
            % modelLoss function.
            [loss,gradients] = dlfeval(@modelLoss,parameters,X,A,T);        
            % Update the network parameters using the Adam optimizer
            [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients, ...
                trailingAvg,trailingAvgSq,iteration,learnRate);
            % Update training progress monitor
            D = duration(0,0,toc(start),Format="hh:mm:ss");
            title("Epoch: " + epoch + ", Elapsed: " + string(D));
            loss = double(loss);L=[L,loss];
            addpoints(lineLossTrain,iteration,loss);
            drawnow;
        end
    end
else
    % Download and unzip the folder containing the pretrained parameters
    zipFile = matlab.internal.examples.downloadSupportFile("nnet","data/parametersHumanActivity_GDN.zip");
    dataFolder = fileparts(zipFile);
    unzip(zipFile,dataFolder);
    % Load the pretrained parameters
    load(fullfile(dataFolder,"parametersHumanActivity_GDN","parameters.mat"))
end

%% Test model
data = load('D:\Dataset\CMAPSS\test_FD001.txt');
RUL = load('D:\Dataset\CMAPSS\RUL_FD001.txt');
ID = data(:,1);
featuresTest = [data(:,7:9),data(:,12:14),data(:,16:20),data(:,22),data(:,25:26)];
featuresTest = normalize(featuresTest,center=muData,scale=sigmaData);
[XTest,ATest,TTest] = processTestData(featuresTest,ID,RUL,windowSize);
dsXTest = arrayDatastore(XTest,IterationDimension=3,ReadSize=miniBatchSize);
dsATest = arrayDatastore(ATest,IterationDimension=3,ReadSize=miniBatchSize);
dsTest = combine(dsXTest,dsATest);% Test data

% Model prediction
YTest = modelPredictions(parameters,dsTest);
figure(2)
plot(TTest)
hold on
plot(YTest)

% Performance evaluation (RMSE & Score)
TTest=TTest';
YTest=gather(extractdata(YTest))';
RMSE = sqrt(mean((TTest-YTest).^2));
Score=[];
diff=YTest-TTest;
for i=1:size(diff,1)
    if diff(i)>0
        score=exp(diff(i)/10)-1;
    else
        score=exp(-diff(i)/13)-1;
    end
    Score=[Score;score];
end
Score=sum(Score);