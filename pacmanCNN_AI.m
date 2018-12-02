function [new_dir,action] = pacmanCNN_AI(pacman,Tuple,epsilon,net)

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
%
%% Output:
% targetSquare: this is the tile where pacman is sent to after this function is done
%
%
%% Nested functions:
% curSquare = findSquare(entity,dir):
% returns the current tile a ghost or pacman (entity) is at right now
%
% possibleMoves = allPossibleMoves(entity):
% returns all possible moves the entity (pacman or ghost) can go to at its current position

%% AI
%% Small Game AI
% Game actions
% Right Down Left Up   Do nothing 
%   1    2    3   4        5
i=5; % Change this to 4 or 5 to choose number of possible actions


if rand>epsilon
    Q=predict(net,Tuple.Frames);
    [~,action]=max(Q);
else
    action=randi(i);
end 

if action~=5
    new_dir=action;
else 
    new_dir=pacman.dir;
end
    
    
%% Nested Functions
    function curSquare = findSquare(entity,dir)
        if dir == 1 || dir == 4
            curSquare = [round(entity.pos(1)-0.45),round(entity.pos(2)-0.45)];
        else
            curSquare = [round(entity.pos(1)+0.45),round(entity.pos(2)+0.45)];
        end
    end

    function possibleMoves = allPossibleMoves(entity)
        curSquare = findSquare(entity,entity.dir);
        possibleMoves = allDirections{curSquare(1),curSquare(2)};
    end
end