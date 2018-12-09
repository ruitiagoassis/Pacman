function [rede_decisao] = criaRedeDecisao(state_memory,q_value_memory,rede_decisao)
%UNTITLED Summary of this function goes here
% Detailed explanation goes here
state_memory = state_memory';
rep = [0,0,0,0,0,0];
q_value_memory=q_value_memory';
rep2 = [0,0,0,0];
if sum(ismember(state_memory,rep,'rows'))==0 && sum(ismember(q_value_memory,rep2,'rows'))==0
    q_value_memory=q_value_memory';
    state_memory = state_memory';
    rede_decisao = newrbe(state_memory,q_value_memory);
    rede_decisao.layers{2}.transferFcn = 'tansig';
end
% Cria Radial Basis net


end

