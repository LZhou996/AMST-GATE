function [XData,Adjacency,TData] = processTrainData(features, ID, windowSize)

maxID = max(ID);
XData = zeros(windowSize,size(features,2));
Adjacency = zeros(size(features,2),size(features,2));
TData = [];

for i = 1 : maxID
    idx = find(ID==i);
    rul = (numel(idx):-1:1)';rul=rul(windowSize:end);idx1 = find(rul>125);rul(idx1) = 125;
    matrix = features(idx,:);
    for j = 1 : size(matrix,1)-windowSize+1
        sample = matrix(j:j+windowSize-1,:);
        XData = cat(3,XData,sample);
        for m = 1 : size(sample,2)
            for n = 1 : size(sample,2)
                cos(m,n)=dot(sample(:,m),sample(:,n))/(norm(sample(:,m))*norm(sample(:,n)));
            end
        end
        idx1 = find(cos<0);
        cos(idx1) = 0;
        idx2 = find(cos>0);
        cos(idx2) = 1;
        Adjacency = cat(3,Adjacency,cos);
    end
    TData = [TData;rul];
end
XData = XData(:,:,2:end);
Adjacency = Adjacency(:,:,2:end);
TData = permute(TData,[2 1]);
end