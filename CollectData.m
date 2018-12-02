d=dir('States\Episode_*.mat');
load(['States\' d(1).name])
dSize=length(Tuples);

XTrain=zeros([size(Tuples(1).Frames) dSize*length(d)]);
YTrain=zeros(dSize*length(d),5);
dSize=0;
for i=1:length(d)
    
    epi=load(['States\' d(i).name]);
    Tuples=epi.Tuples;

    if isempty(Tuples(end).action)
        Tuples(end)=[];
    end
    for j=1:length(Tuples)
        XTrain(:,:,:,dSize+j)=Tuples(j).Frames;
        YTrain(dSize+j,:)=Tuples(j).Q;
    end
    dSize=dSize+length(Tuples);
end

save('TrainingDATA.mat','XTrain','YTrain');