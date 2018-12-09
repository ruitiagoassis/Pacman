function accao_1 = pacmanAI(pacman,enemies,allDirections,coins,pills,lives)
% targetSquare = pacmanAI(pacman,enemies,allDirections)

%% Input:
% pacman: struct-array with all of the pacman information
% pacman.pos: current position [x,y]
%
% enemies: struct-array with all of the ghosts information
% enemies(1).pos: current position of ghost No.1 [x,y]
% enemies(2).pos: current position of ghost No.2 [x,y]
% enemies(3).pos: current position of ghost No.3 [x,y]
% enemies(4).pos: current position of ghost No.4 [x,y]
%
% allDirections: cell-array with all possible moves for each tile in the game
%
% coins: struct-array with all of the coins information
%
% pills: struct-array with all of the pills information

%% Global variables
global net_decisao
global reward
global max_reward
global first_game_over
global versao
global stacked_frame
global q_value_memoria

%% Persistent Variables

persistent q_value
persistent q_value_anterior




persistent accao
persistent accao_anterior
persistent alfa % Learning Rate
persistent gamma % Discount Rate
persistent time_memory


if isempty(accao_anterior)
    accao_anterior = 1;
end
if isempty(q_value)
    
    q_value =[0;0;0;0];
end

accao_anterior = accao;




if isempty(time_memory)
    tic
    time_memory = 0;
end

if time_memory
    tic
    time_memory = 0;
end

alfa = 0.1;
gamma = 0.9;

q_value_anterior = q_value;


if isempty(q_value_anterior)
    q_value_anterior=[0;0;0;0];
end

%% Rewards

% Calcular distância à coin mais próxima
% Verifica se a reward se deve a comer coin ou a morrer,
% Caso seja uma destas duas queremos que a rede atualize com a reward da
% coin ou da morte, antes de voltar a fornecer reward baseado na distância
% à próxima moeda
%
if reward ~= 0.5*max_reward && reward ~= -max_reward && reward ~= 0.6*max_reward && reward ~= max_reward
    reward = -0.01*max_reward;
end


%% Implementação



% É feito o mapeamento do local do pacman, fantasmas e moedas através da rede neuronal que
% representa o mapa de jogo - net_mapa.
% O mapeamento feito é seguidamente introduzido na segunda rede neuronal - a
% net_decisao - que vai calcular os Q values esperados para aquela
% decisão



if ~first_game_over
    accao = randi([1 4]);
    accao_1 = accao;
elseif size(stacked_frame,3)==4
    
    q_value = predict(net_decisao,stacked_frame);
    
    
    % Variação do fator de exploração com o número de vidas restante
    
    random_or_net = randi([0 99]);
    %
    
    if random_or_net <5
        
        accao = randi([1 4]);
        accao_1 = accao;
        % Toma acção random entre andar numa das direções ou manter-se na mesma
        % direcção
    else
        % Usa a rede neuronal para tomar a decisão
        % Escolhe o maior Q value que é retirado da rede
        % Acção é identificada da seguinte forma
        % Direita -> 1
        % Baixo -> 2
        % Esquerda -> 3
        % Cima -> 4
        % Permanecer na mesma direcção -> 5
        
        accao = find(q_value == max(q_value),1);
        accao_1 = accao;
        
        
    end
    
    q_value_novo = q_value_anterior(accao_anterior) + alfa * ( reward + gamma * q_value(accao_anterior) -q_value_anterior(accao_anterior));
    
    q_value_anterior(accao_anterior)=q_value_novo;
    
    
    
    
    q_value_memoria(:,:,1)=q_value_anterior(1);
    q_value_memoria(:,:,2)=q_value_anterior(2);
    q_value_memoria(:,:,3)=q_value_anterior(3);
    q_value_memoria(:,:,4)=q_value_anterior(4);
else
    accao_1 = accao_anterior;
end

% Reset à reward para poder ser alterada após a decisão, caso a acção
% anterior tenha sido comer uma coin ou morrer

if reward == 0.5*max_reward || reward == -max_reward || reward == 0.6*max_reward || reward == max_reward
    reward =0;
    
end

pause(0.000000001)

end