function net = treinarRede(net)


global q_value_memoria
tamanho_memoria=(size(dir('Frames'),1))-2;
random_batch = randi([1 tamanho_memoria],[1 32]);


if isempty(net)
    state = zeros(64,64,4,tamanho_memoria);
    for i=1:1:tamanho_memoria
        load(sprintf('Frames\\stacked_frame_%i',i),'stacked_frame')
        state(:,:,:,i)=stacked_frame;
        q_value_temporario(:,:,1)=0;
        q_value_temporario(:,:,2)=1;
        q_value_temporario(:,:,3)=2;
        q_value_temporario(:,:,4)=3;
        q_value(:,:,:,i)=q_value_temporario;
    end
else
    state = zeros(64,64,4,32);
    for i=1:1:32
        load(sprintf('Frames\\stacked_frame_%i',random_batch(i)),'stacked_frame')
        state(:,:,:,i)=stacked_frame;
        load(sprintf('Q_Values\\q_value_memoria_%i',random_batch(i)),'q_value_memoria')
        q_value(:,:,:,i)=q_value_memoria;
    end
end

% Create layer
if isempty(net)
    load('Layers_rede','Layer')
    
    options = trainingOptions('sgdm', ...
        'MaxEpochs',10,...
        'InitialLearnRate',1e-4, ...
        'ExecutionEnvironment','parallel',...
        'Verbose',false);
    net = trainNetwork(state,q_value,Layer,options);
    
else
    
    Layer = net.Layers;
    
    options = trainingOptions('sgdm', ...
        'MaxEpochs',10,...
        'InitialLearnRate',1e-4, ...
        'ExecutionEnvironment','parallel',...
        'Verbose',false);
    net = trainNetwork(state,q_value,Layer,options);
end




end

