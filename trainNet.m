function CNN=trainNet(CNN,Tuples)
if isempty(Tuples(end).action)
    Tuples(end)=[];
end
%d=dir('States\Episode_*.mat');
% idxFiles=randperm(length(d),5);
dSize=length(Tuples);

X=zeros([size(Tuples(1).Frames) dSize]);
Y=zeros(dSize,5);
for i=1:dSize
X(:,:,:,i)=Tuples(i).Frames;
Y(i,:)=Tuples(i).Q;
end

options = trainingOptions('sgdm', ...
    'MiniBatchSize',20,...
    'InitialLearnRate',0.05, ...
    'Verbose',false, ...
    'Plots','none',...   % 'training-progress' or 'none'
    'MaxEpochs',20,...
    'Shuffle','every-epoch',...
    'L2Regularization',0.0005,...
    'Momentum',0.9,...
    'ExecutionEnvironment','auto');
    %'CheckpointPath',checkpointPath);
    
lgraph = createLgraphUsingConnections(CNN.net.Layers,CNN.net.Connections);

[CNN.net,CNN.trinfo]=trainNetwork(X,Y,lgraph,options);
end
function lgraph = createLgraphUsingConnections(layers,connections)
% lgraph = createLgraphUsingConnections(layers,connections) creates a layer
% graph with the layers in the layer array |layers| connected by the
% connections in |connections|.

lgraph = layerGraph();
for i = 1:numel(layers)
    lgraph = addLayers(lgraph,layers(i));
end

for c = 1:size(connections,1)
    lgraph = connectLayers(lgraph,connections.Source{c},connections.Destination{c});
end

end



