function [outputFeatures,attentionScore] = graphAttention(inputFeatures,adjacency,weights)
linearWeights = weights.linear;
attentionWeights = weights.attention;

% Compute linear transformation of input features
value = pagemtimes(linearWeights, inputFeatures);

% Compute attention coefficients
query = pagemtimes(attentionWeights(1, :), value);
key = pagemtimes(attentionWeights(2, :), value);

attentionCoefficients = query + permute(key,[2, 1, 3]);
attentionCoefficients = leakyrelu(attentionCoefficients,0.2);

% Compute masked attention coefficients
mask = -10e9 * (1 - adjacency);
attentionCoefficients = (attentionCoefficients .* adjacency) + mask;

% Compute normalized masked attention coefficients
attentionScore = softmax(attentionCoefficients,DataFormat = "UCB");

% Normalize features using normalized masked attention coefficients
outputFeatures = pagemtimes(value, attentionScore);
end