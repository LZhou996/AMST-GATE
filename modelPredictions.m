function Y = modelPredictions(parameters,ds)
arguments
    parameters
    ds
end

Y = [];

reset(ds)
while hasdata(ds)

    data = read(ds);

    X = data(:,1); X = cat(3,X{:});

    A = data(:,2); A = cat(3,A{:});

    miniBatchPred = model(parameters,X,A);
    Y = cat(2,Y,miniBatchPred);

end
end