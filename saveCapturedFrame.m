function saveCapturedFrame(versao)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

global q_value_memoria
global stacked_frame
persistent contagem


if isempty(contagem)
    contagem = 1;
end
if contagem == 500
    contagem = 1;
end

frame_1 = frame2im(getframe);

frame_1 = rgb2gray(frame_1);

frame_1(395:end,:)=[];

frame_1(1:205,:)=[];

frame_1(:,415:end)=[];

frame_1(:,1:220)=[];

frame_1 = imresize(frame_1,[64 64]);

stacked_frame(:,:,mod(versao,4)+1)=frame_1;

if size(stacked_frame,3)==4
    save(sprintf('Frames\\stacked_frame_%i',contagem),'stacked_frame')
    save(sprintf('Q_Values\\q_value_memoria_%i',contagem),'q_value_memoria')
    contagem = contagem + 1;
end
end

