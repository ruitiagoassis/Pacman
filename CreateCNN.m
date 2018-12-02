% Input and Output Sizes
iSize=[62,130,4];   %[124,260,4];
oSize=5;

%     % Pre-trained AlexNet 
%     net=alexnet;
%     layers=net.Layers;
%     layers(1)= imageInputLayer(iSize);
%     layers(end)=[];
%     layers(end-1)=fullyConnectedLayer(oSize);
%     layers(end)=regressionLayer('Name','Q-Values');


    % Customized Net
%     layers = [ ...
%         imageInputLayer(iSize)
%         convolution2dLayer(12,25)
%         batchNormalizationLayer
%         maxPooling2dLayer(2,'Stride',3)
%         reluLayer
%         convolution2dLayer(12,25) % ,'WeightLearnRateFactor',1/2)
%         batchNormalizationLayer
%         maxPooling2dLayer(2,'Stride',3)
%         reluLayer
%         crossChannelNormalizationLayer(4,'K',1)
%         fullyConnectedLayer(oSize)
%         regressionLayer('Name','Q-Values')];
load('CNNets\SqueezeNet.mat');   
    
lgraph = createLgraphUsingConnections(layers,connections);

% Choosing Validation Data randomly
load('TrainingDATA.mat')


dSize=length(YTrain);
idx = randperm(size(XTrain,4),floor(dSize*0.2));
XValidation = XTrain(:,:,:,idx);
XTrain(:,:,:,idx) = [];
YValidation = YTrain(idx,:);
YTrain(idx,:) = [];

% Training Options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',80,...
    'InitialLearnRate',0.1, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'MaxEpochs',40,...
    'Shuffle','every-epoch',...
    'ValidationData',{XValidation,YValidation},...
    'ValidationFrequency',15,...
    'ValidationPatience',7,...
    'L2Regularization',0.0005,...
    'Momentum',0.9,...
    'ExecutionEnvironment','parallel');
%    'CheckpointPath',checkpointPath);

[net,trinfo]=trainNetwork(XTrain,YTrain,lgraph,options);

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

function c = subgraphConnections(src,dest)
if numel(src) > 1
    dest_name_root = [dest.Name '/in'];
    tmp = cell(numel(src),2);
    for k = 1:numel(src)
        tmp{k,1} = src(k).Name;
        tmp{k,2} = [dest_name_root num2str(k)];
    end
elseif numel(dest) > 1
    tmp = cell(numel(dest),2);
    for k = 1:numel(dest)
        tmp{k,1} = src.Name;
        tmp{k,2} = dest(k).Name;
    end
end
c = table(tmp(:,1),tmp(:,2),'VariableNames',{'Source','Destination'});
end