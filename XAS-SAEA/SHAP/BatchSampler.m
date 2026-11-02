function val=BatchSampler(X,batchsize)
    L=size(X,1);
    num=floor(L./batchsize);
    val=cell(num,1);
    for i=1:num
        l=(i-1)*batchsize+1;
        r=i*batchsize;
        val{i}=X(l:r,:);

    end
end