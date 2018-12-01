function [mem,net] = updateAndTrain(mem,net)
%% Update positions where pacman has been


for i=1:size(mem,1)
    if mem(i,:) ~= ([0 0])
        rep=mem(i,:);
        instancias = find(ismember(mem,rep,'rows'));
        mem(instancias(2:end),:)=0;
    end
end

rep = [0 0];
instancias = find(ismember(mem,rep,'rows'));
mem(instancias(1:end),:)=[];
%
%
% %% Train network
%
% mem=mem';
% % net = configure(net,mem);
% % net = train(net,mem);
% mem = mem';

end

