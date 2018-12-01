function accao_1 = pacmanAI(pacman,enemies,allDirections,coins,pills)
% targetSquare = pacmanAI(pacman,enemies,allDirections)
%
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
global net_mapa
global reward
global max_reward
global first_game_over

%% Persistent Variables
persistent q_value
persistent q_value_anterior
persistent estado_anterior
% persistent distancia_anterior_proxima_moeda
% persistent distancia_minima_fantasmas_anterior
persistent accao
persistent accao_anterior
persistent alfa % Learning Rate
persistent gamma % Discount Rate

% if isempty(distancia_minima_fantasmas_anterior)
%     distancia_minima_fantasmas_anterior = 999;
% end
alfa = 0.1;
gamma = 0.9;
accao_anterior = accao;
q_value_anterior = q_value;

%% Rewards

% Calcular distância à coin mais próxima

% distancia_moeda = zeros(1,size(coins.data,1));
% for i=1:size(coins.data,1)
%     distancia_moeda(i) = abs(pacman.pos(1)-coins.data(i,1))+abs(pacman.pos(2)-coins.data(i,2));
% end
% 
% distancia_proxima_moeda = min(distancia_moeda);

% Verificar se o Pacman se está a aproximar de fantasmas.
% distancia_fantasmas = zeros(1,4);
% for i=1:4
%     distancia_fantasmas(i) = abs(pacman.pos(1)-enemies(i).pos(1))+abs(pacman.pos(2)-enemies(i).pos(2));
% end
% 
% distancia_minima_fantasmas = min(distancia_fantasmas);
% fantasma_mais_proximo = find(distancia_minima_fantasmas == min(distancia_fantasmas));
% Verifica se a distância à moeda mais próxima aumentou ou diminuiu

% Verifica se a reward se deve a comer coin ou a morrer,
% Caso seja uma destas duas queremos que a rede atualize com a reward da
% coin ou da morte, antes de voltar a fornecer reward baseado na distância
% à próxima moeda
% 
if reward ~= 0.5*max_reward && reward ~= -max_reward
        reward = -0.1*max_reward;
end
% 
% distancia_anterior_proxima_moeda = distancia_proxima_moeda;




%% Implementação



% É feito o mapeamento do local do pacman, fantasmas e moedas através da rede neuronal que
% representa o mapa de jogo - net_mapa.
% O mapeamento feito é seguidamente introduzido na segunda rede neuronal - a
% net_decisao - que vai calcular os Q values esperados para aquela
% decisão

% Mapeia a posição do pacman
pos_pacman = sim(net_mapa,pacman.pos');
% Mapeia a posição dos fantasmas
pos_fantasmas = sim(net_mapa,[enemies(1).pos',enemies(2).pos',enemies(3).pos',enemies(4).pos']);
% Mapeia a posição das moedas
pos_moedas = sim(net_mapa,coins.data');
% Posição das pills
pos_pills = sim(net_mapa,pills.data');
% Agregamento dos vários vetores de posicao
% Legenda
% Moedas não ativas -> 0
% Moedas ativas -> 1
% Pills não ativas -> 0
% Pills ativas -> 2
% Pacman -> 4
% Fantasmas agressivos -> 5
% Fantasmas vulneráveis -> 3
% Fantasmas na caixa -> 0
% Fantasmas em modo olhos -> 0
% Identifica se fantasmas podem ser comidos ou não
estado_fantasmas = zeros(1,4);
for i=1:4
    if enemies(i).status == 0
        estado_fantasmas(i)=0;
    elseif enemies(i).status == 1
        estado_fantasmas(i)=5;
    elseif enemies(i).status == 2
        estado_fantasmas(i)=3;
    elseif enemies(i).status == 3
        estado_fantasmas(i)=0;
    end
end

% if distancia_minima_fantasmas < 6
%     if distancia_minima_fantasmas < distancia_minima_fantasmas_anterior &&  estado_fantasmas(fantasma_mais_proximo)==6
%         reward = -0.6*max_reward;
%     end
% end
% distancia_minima_fantasmas_anterior = distancia_minima_fantasmas;

pos_fantasmas = pos_fantasmas.*estado_fantasmas;
estado = [4*pos_pacman,pos_fantasmas,pos_moedas,3*pos_pills];
estado = max(estado,[],2);

if isempty(estado_anterior)
    estado_anterior = estado;
end
if ~first_game_over
    accao = randi([1 5]);
    accao_1 = accao;
else
    q_value = sim(net_decisao,estado);
    
    random_or_net = randi([0 1]);
    
    if ~random_or_net
        
        accao = randi([1 5]);
        accao_1 = accao;
        
        aprendizagem(max(q_value));
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
        aprendizagem(max(q_value));
    end
end
estado_anterior = estado;
% Reset à reward para poder ser alterada após a decisão, caso a acção
% anterior tenha sido comer uma coin ou morrer
if reward == 0.5*max_reward || reward == -max_reward
    reward =0;
    
end
pause(0.0000001)
    function aprendizagem(max_q_value)
        
        q_value_melhor = updateQValue(q_value_anterior(accao_anterior),max_q_value);
        
        q_value_anterior(accao_anterior)=q_value_melhor;
        
        net_decisao = adapt(net_decisao,estado_anterior,q_value_anterior);
        
    end

    function q_value_novo = updateQValue(q_value_anterior,max_q_value)
        
        q_value_novo = q_value_anterior + alfa*(reward + gamma * max_q_value - q_value_anterior);
        
        
    end
end