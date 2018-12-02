for i=1:2
    figure(i)
    for j=1:12
        subplot(3,4,j)
        imagesc(Tuples(j+(i-1)*12).Frames(:,:,1));
    end
end