function accao_1 = pacmanAI(pacman,enemies,allDirections,coins,pills,lives)
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
global net_decisao_1
global net_decisao_2
global net_mapa
global reward
global max_reward
global first_game_over
global state_memory_1
global q_value_memory_1
global state_memory_2
global q_value_memory_2
global versao

%% Persistent Variables
persistent q_value_1
persistent q_value_2

persistent q_value_anterior_1
persistent q_value_anterior_2

persistent posicao_pacman_anterior
persistent distancia_anterior_proxima_moeda
persistent distancia_minima_fantasmas_anterior

persistent estado_anterior
persistent accao
persistent accao_anterior
persistent alfa % Learning Rate
persistent gamma % Discount Rate
persistent time_memory
persistent counter_1
persistent counter_2

if isempty(accao_anterior)
    accao_anterior = 5;
    accao_1 = 5;
end

if isempty(posicao_pacman_anterior)
    
    posicao_pacman_anterior =  [0 0];
end
if isempty(distancia_anterior_proxima_moeda)
    distancia_anterior_proxima_moeda = 99;
end
if mod(versao,12)
    accao_1 = accao_anterior;
else
    accao_anterior = accao;
end


if isempty(counter_1)
    counter_1 = 10000;
end

if isempty(counter_2)
    counter_2 = 10000;
end

if isempty(time_memory)
    tic
    time_memory = 0;
end

if time_memory
    tic
    time_memory = 0;
end
if isempty(distancia_minima_fantasmas_anterior)
    distancia_minima_fantasmas_anterior = 999;
end
alfa = 0.1;
gamma = 0.9;


q_value_anterior_1 = q_value_1;
q_value_anterior_2 = q_value_2;

if isempty(q_value_anterior_1)
    q_value_anterior_1 = [0;0;0;0;0];
end

if isempty(q_value_anterior_2)
    q_value_anterior_2 = [0;0;0;0;0];
end
%% Rewards

% Calcular distância à coin mais próxima

distancia_moeda = zeros(1,size(coins.data,1));
for i=1:size(coins.data,1)
    distancia_moeda(i) = abs(pacman.pos(1)-coins.data(i,1))+abs(pacman.pos(2)-coins.data(i,2));
end
%
distancia_proxima_moeda = min(distancia_moeda);

% Verificar se o Pacman se está a aproximar de fantasmas.
distancia_fantasmas = zeros(1,1);
for i=1:1
    distancia_fantasmas(i) = abs(pacman.pos(1)-enemies(i).pos(1))+abs(pacman.pos(2)-enemies(i).pos(2));
end
%
distancia_minima_fantasmas = min(distancia_fantasmas);
% fantasma_mais_proximo = find(distancia_minima_fantasmas == min(distancia_fantasmas));
% Verifica se a distância à moeda mais próxima aumentou ou diminuiu




% Inputs

% % distância à moeda mais próxima
% distancia_proxima_moeda;
% % distância ao fantasma mas próximo
% distancia_minima_fantasmas;
% % Se pode comer fantasma
% % 1 ou 0
% % Direção atual
% pacman.dir;
% curSquare = findSquare(pacman,pacman.dir);
% pode_virar = allDirections{curSquare(1),curSquare(2)};
% % pode virar direita
% if isempty(find(pode_virar==1,1))
%     input_direita = 0 ;
% else
%     input_direita = 1;
% end
% % pode virar baixo
% if isempty(find(pode_virar==2,1))
%     input_baixo = 0 ;
% else
%     input_baixo = 1;
% end
% % pode virar esquerda
% if isempty(find(pode_virar==3,1))
%     input_esquerda = 0 ;
% else
%     input_esquerda = 1;
% end
% % pode virar cima
% if isempty(find(pode_virar==4,1))
%     input_cima = 0 ;
% else
%     input_cima = 1;
% end
%
% input_direcoes = [ input_direita;input_baixo;input_esquerda;input_cima];
%
% estado = [pacman.pos(1);...
%     pacman.pos(2);...
%     enemies(1).pos(1);...
%     enemies(1).pos(2);...
%     input_direita;...
%     input_baixo;...
%     input_esquerda;...
%     input_cima];


% Verifica se a reward se deve a comer coin ou a morrer,
% Caso seja uma destas duas queremos que a rede atualize com a reward da
% coin ou da morte, antes de voltar a fornecer reward baseado na distância
% à próxima moeda
%
if reward ~=0.5*max_reward && reward ~= -max_reward && reward ~= 0.6*max_reward && reward ~= max_reward
    if distancia_anterior_proxima_moeda> distancia_proxima_moeda
        reward = 0.1*max_reward;
    elseif posicao_pacman_anterior == pacman.pos
        reward = -max_reward;
    else
        reward = -0.05*max_reward;
    end
end



distancia_minima_fantasmas_anterior = distancia_minima_fantasmas;
% posicao_pacman_anterior = pacman.pos;
%
distancia_anterior_proxima_moeda = distancia_proxima_moeda;
%
%
%

%% Implementação



% É feito o mapeamento do local do pacman, fantasmas e moedas através da rede neuronal que
% representa o mapa de jogo - net_mapa.
% O mapeamento feito é seguidamente introduzido na segunda rede neuronal - a
% net_decisao - que vai calcular os Q values esperados para aquela
% decisão

% Mapeia a posição do pacman
pos_pacman = sim(net_mapa,pacman.pos');
% Mapeia a posição dos fantasmas
pos_fantasmas = sim(net_mapa,[enemies(1).pos']);
% Mapeia a posição das moedas
pos_moedas = sim(net_mapa,coins.data');
% Posição das pills
% pos_pills = sim(net_mapa,pills.data');
% % Agregamento dos vários vetores de posicao
% % Legenda
% % Moedas não ativas -> 0
% % Moedas ativas -> 1
% % Pills não ativas -> 0
% % Pills ativas -> 2
% % Pacman -> 4
% % Fantasmas agressivos -> 5
% % Fantasmas vulneráveis -> 3
% % Fantasmas na caixa -> 0
% % Fantasmas em modo olhos -> 0
% % Identifica se fantasmas podem ser comidos ou não
estado_fantasmas = zeros(1,1);
for i=1:1
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
%
%
%
pos_fantasmas = pos_fantasmas.*estado_fantasmas;
estado = [4*pos_pacman,pos_fantasmas,pos_moedas];
estado = max(estado,[],2);
%
if isempty(estado_anterior)
    estado_anterior = estado;
end

if ~first_game_over
    accao = randi([1 4]);
    accao_1 = accao;
else
    q_value_1 = sim(net_decisao_1,estado);
    q_value_2 = sim(net_decisao_2,estado);
    
    possivel_accao = [q_value_1,q_value_2];
    possivel_accao = max(possivel_accao,[],2);
    % Variação do fator de exploração com o número de vidas restante
    
    random_or_net = randi([0 99]);
    %
    
    if random_or_net <50
        
        accao = randi([1 4]);
        if ~mod(versao,12)
            accao_1 = accao;
        end
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
        
        accao = find(possivel_accao == max(possivel_accao),1);
        if ~mod(versao,12)
            accao_1 = accao;
        end
        
        
    end
    
    
    net_one_or_two = randi([0 1]);
    
    if net_one_or_two
        
        q_value_anterior_1 = aprendizagem(net_decisao_2,q_value_1,estado,q_value_anterior_1);
        q_value_memory_1(:,counter_1)=q_value_anterior_1;
        state_memory_1(:,counter_1)=estado_anterior;
        counter_1 = counter_1 -1;
        
        if counter_1 <1
            
            counter_1 = 10000;
            
        end
        
    elseif ~net_one_or_two
        
        q_value_anterior_2 = aprendizagem(net_decisao_1,q_value_2,estado,q_value_anterior_2);
        q_value_memory_2(:,counter_2)=q_value_anterior_2;
        state_memory_2(:,counter_2)=estado_anterior;
        
        counter_2 = counter_2 -1;
        
        if counter_2 <1
            
            counter_2 = 10000;
            
        end
        
    end
    time = toc-6;
    
    if time >0.1
        time = 0;
    end
    
    if -0.1<time && time<0.1
        sample = randi([1 10000],[1 300]);
        if net_one_or_two
            
            net_decisao_1 = train(net_decisao_1,state_memory_1(:,sample),q_value_memory_1(:,sample),'UseParallel','yes');
            time_memory = 1;
            
        elseif ~net_one_or_two
            
            
            net_decisao_2 = train(net_decisao_2,state_memory_2(:,sample),q_value_memory_2(:,sample),'UseParallel','yes');
            time_memory = 1;
        end
    end
    
    estado_anterior = estado;
end

% Reset à reward para poder ser alterada após a decisão, caso a acção
% anterior tenha sido comer uma coin ou morrer

if reward == 0.5*max_reward || reward == -max_reward || reward == 0.6*max_reward || reward == max_reward || reward == -0.6*max_reward
    reward =0;
    
end
pause(0.000000001)

    function q_value_anterior = aprendizagem(net_decisao,q_value,estado,q_value_anterior)
        q_value_novo = updateQValue(net_decisao,q_value,estado,q_value_anterior(accao_anterior));
        q_value_anterior(accao_anterior)=q_value_novo;
    end


    function q_value_novo = updateQValue(net_decisao,q_value,estado,q_value_accao_anterior)
        
        indice_max_q_value = q_value == max(q_value);
        
        new_max_q_value = sim(net_decisao,estado);
        
        
        q_value_novo = q_value_accao_anterior + alfa * ( reward + gamma * (new_max_q_value(indice_max_q_value)) -q_value_accao_anterior);
        
    end



end