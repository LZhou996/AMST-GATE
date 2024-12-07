function [XData,Adjacency,TData] = processTestData1(features, ID, RUL, windowSize)
maxID = max(ID);
XData = zeros(windowSize,size(features,2),maxID);
Adjacency = zeros(size(features,2),size(features,2),maxID);
for i = 1 : maxID
    idx = find(ID==i);
    sample = features(idx(end-windowSize+1):idx(end),:);
    XData(:,:,i) = sample;
    for m = 1 : size(sample,2)
        for n = 1 : size(sample,2)
                cos(m,n)=dot(sample(:,m),sample(:,n))/(norm(sample(:,m))*norm(sample(:,n)));
        end
    end
        idx1 = find(cos<0);
        cos(idx1) = 0;
        idx2 = find(cos>0);
        cos(idx2) = 1;
        Adjacency(:,:,i) = cos;
end
RUL = RUL+1;
idx3 = find(RUL>125);
RUL(idx3) = 125;
TData = RUL';
end