% 
% d=dir('States\Episode_*.mat');
% 
% idxEpi=randperm(length(d),ceil(length(d)*0.7));
% data=load(['States\' d(idxEpi(1)).name]);
% trainT=data.Tuples;
% idxEpi(1)=[];
% 
% for i=idxEpi
%     data=load(['States\' d(i).name]);
%     idx=randperm(length(data.Tuples),ceil(length(data.Tuples)*0.5));
%     trainT=[trainT;data.Tuples(idx)];
% end
% 
% XTrain=zeros([size(trainT(1).Frames) length(trainT)]);
% YTrain=zeros([size(XTrain,4) length(trainT(1).Q)]);
% 
% for i=1:length(trainT)
%     XTrain(:,:,:,i)=trainT(i).Frames;
%     YTrain(i,:)=trainT(i).Q;
% end
% 
% dSize=length(YTrain);
% idx = randperm(dSize,floor(dSize*0.2));
% XValidation = XTrain(:,:,:,idx);
% YValidation = YTrain(idx,:);
% idx=sort(idx,'descend');
% for i=idx
% XTrain(:,:,:,i) = [];
% YTrain(i,:) = [];
% end
% 
% options = trainingOptions('sgdm', ...
%     'MiniBatchSize',80,...
%     'InitialLearnRate',0.01, ...
%     'Verbose',false, ...
%     'Plots','training-progress',...
%     'MaxEpochs',150,...
%     'Shuffle','every-epoch',...
%     'ValidationData',{XValidation,YValidation},...
%     'ValidationFrequency',15,...
%     'ValidationPatience',7,...
%     'L2Regularization',0.0005,...
%     'Momentum',0.9,...
%     'ExecutionEnvironment','parallel');

% CNN=load('CNNets\Current_Net.mat');
% layers=CNN.net.Layers;
% connections=CNN.net.Connections;
% layers(2).WeightLearnRateFactor=1/3;layers(2).WeightL2Factor=1/3;layers(2).BiasLearnRateFactor=1/3;
lgraph = createLgraphUsingConnections(layers,connections);
[net,trinfo]=trainNetwork(XTrain,YTrain,lgraph,options);

d=dir('CNNets\Net_*.mat');
NNet=length(d)+1;
save(['CNNets\Net_' num2str(NNet) '.mat'],'net','trinfo');
save('CNNets\Current_Net.mat','net','trinfo');
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


