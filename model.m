function [Y,attentionScores] = model(parameters,X,A)

adjacency = A;

% 1st AMS-GAT
% Self-attention mechanism
weights = parameters.graphattn1A.weights;
[outputFeatures1A,attentionScores] = graphAttention(X,adjacency,weights);
outputFeatures1A = relu(outputFeatures1A);%ReLU

% ExpSparse self-attention mechanism
weights = parameters.graphattn1B.weights;
[outputFeatures1B,attentionScores] =ExpgraphAttention(X,adjacency,weights,4);
outputFeatures1B = relu(outputFeatures1B);%ReLU

% ExpSparse self-attention mechanism
weights = parameters.graphattn1C.weights;
[outputFeatures1C,attentionScores] = ExpgraphAttention(X,adjacency,weights,8);
outputFeatures1C = relu(outputFeatures1C);%ReLU

% GSAM
s1 = mean(outputFeatures1A,2);
s2 = mean(outputFeatures1B,2);
s3 = mean(outputFeatures1C,2);
s = cat(2,s1,s2,s3);
sA = mean(s); sA = permute(sA,[2 1 3]);
sB = max(s); sB = permute(sB,[2 1 3]);

weights = parameters.conv1A.Weights;
bias = parameters.conv1A.Bias;
sA = dlconv(sA,weights,bias,Padding="same",DataFormat='SSB');
sA = relu(sA);
weights = parameters.conv1B.Weights;
bias = parameters.conv1B.Bias;
sA = dlconv(sA,weights,bias,Padding="same",DataFormat='SSB');

weights = parameters.conv1A.Weights;
bias = parameters.conv1A.Bias;
sB = dlconv(sB,weights,bias,Padding="same",DataFormat='SSB');
sB = relu(sB);
weights = parameters.conv1B.Weights;
bias = parameters.conv1B.Bias;
sB = dlconv(sB,weights,bias,Padding="same",DataFormat='SSB');

w1 = sigmoid(sA+sB);

outputFeatures = w1(1).*outputFeatures1A + w1(2).*outputFeatures1B + w1(3).*outputFeatures1C;

% 2nd AMS-GAT
% Self-attention mechanism
weights = parameters.graphattn2A.weights;
[outputFeatures2A,attentionScores] = graphAttention(outputFeatures,adjacency,weights);
outputFeatures2A = relu(outputFeatures2A);%ReLU

% ExpSparse self-attention mechanism
weights = parameters.graphattn2B.weights;
[outputFeatures2B,attentionScores] =ExpgraphAttention(outputFeatures,adjacency,weights,4);
outputFeatures2B = relu(outputFeatures2B);%ReLU

% ExpSparse self-attention mechanism
weights = parameters.graphattn2C.weights;
[outputFeatures2C,attentionScores] = ExpgraphAttention(outputFeatures,adjacency,weights,8);
outputFeatures2C = relu(outputFeatures2C);%ReLU

% GSAM
s1 = mean(outputFeatures2A,2);
s2 = mean(outputFeatures2B,2);
s3 = mean(outputFeatures2C,2);
s = cat(2,s1,s2,s3);
sA = mean(s); sA = permute(sA,[2 1 3]);
sB = max(s); sB = permute(sB,[2 1 3]);

weights = parameters.conv2A.Weights;
bias = parameters.conv2A.Bias;
sA = dlconv(sA,weights,bias,Padding="same",DataFormat='SSB');
sA = relu(sA);
weights = parameters.conv2B.Weights;
bias = parameters.conv2B.Bias;
sA = dlconv(sA,weights,bias,Padding="same",DataFormat='SSB');

weights = parameters.conv2A.Weights;
bias = parameters.conv2A.Bias;
sB = dlconv(sB,weights,bias,Padding="same",DataFormat='SSB');
sB = relu(sB);
weights = parameters.conv2B.Weights;
bias = parameters.conv2B.Bias;
sB = dlconv(sB,weights,bias,Padding="same",DataFormat='SSB');

w2 = sigmoid(sA+sB);

outputFeatures = w2(1).*outputFeatures2A + w2(2).*outputFeatures2B + w2(3).*outputFeatures2C;

% ABiTCN
% Positive residual block
outputFeatures1 = reshape(outputFeatures,[size(outputFeatures,1) size(outputFeatures,2) 1 size(outputFeatures,3)]);% Reshape
weights = parameters.conv11.Weights;
bias = parameters.conv11.Bias;
Y11 = dlconv(outputFeatures1,weights,bias,Padding="same",DataFormat='SSCB',DilationFactor=1);
Y11 = relu(Y11);

weights = parameters.conv12.Weights;
bias = parameters.conv12.Bias;
Y12 = dlconv(Y11,weights,bias,Padding="same",DataFormat='SSCB',DilationFactor=2);
Y12 = relu(Y12);

weights = parameters.conv13.Weights;
bias = parameters.conv13.Bias;
Y13 = dlconv(outputFeatures1,weights,bias,Padding="same",DataFormat='SSCB');

Y13 = Y12 + Y13;

% Reverse residual block
outputFeatures2 = flip(outputFeatures,3);% Flip
outputFeatures2 = reshape(outputFeatures2,[size(outputFeatures,1) size(outputFeatures,2) 1 size(outputFeatures,3)]);
weights = parameters.conv21.Weights;
bias = parameters.conv21.Bias;
Y21 = dlconv(outputFeatures2,weights,bias,Padding="same",DataFormat='SSCB',DilationFactor=1);
Y21 = relu(Y21);

weights = parameters.conv22.Weights;
bias = parameters.conv22.Bias;
Y22 = dlconv(Y21,weights,bias,Padding="same",DataFormat='SSCB',DilationFactor=2);
Y22 = relu(Y22);

weights = parameters.conv23.Weights;
bias = parameters.conv23.Bias;
Y23 = dlconv(outputFeatures2,weights,bias,Padding="same",DataFormat='SSCB');

Y23 = Y22 + Y23;

% Bidirectional temporal feature fusion
alpha = parameters.alpha;
alpha = sigmoid(alpha);
Y = alpha.*Y13 + (1-alpha).*Y23;

% Fully Connected layer
weights = parameters.fc.weights;
bias = parameters.fc.bias;
Y = fullyconnect(Y,weights,bias,DataFormat="SSCB");

end