function [obj,shap] = FastShap(Problem,sur,X)
% FastShap model结构及训练
%input
    %Problem问题，提供num_features
    %sur 模型dacefit
    %X 输入数据
    %dacefit预测val,num_features,groups
%output
    %调用：forward(FastShap,X);
 %参考
 %https://github.com/neurips3408/fastshap
 
    %normalization to do 
    Epoch = 40;
    numObservations = size(X,1);
    batch_size=10;
    num_features=Problem.D;
    num_players=num_features;
    l2Regularization = 0.001; % %lbdpara
    lambda=0.1;
    learnRate = 0.1;
    gradientDecayFactor = 0.9;
    squaredGradientDecayFactor = 0.999;
    num_samples=1;%代调
    len1=batch_size * num_samples;
    numIterationsPerEpoch = floor(numObservations./batch_size);
    averageGrad = [];
    averageSqGrad = [];
    numIterations = Epoch * numIterationsPerEpoch;
    %% 训练模型
    explainer=Explainer(Problem);
    net=explainer;
    X1=X;
    grand=calculate_grand_coalition(X1,sur, len1);%这个很重要-待确定计算联盟值
    batchnum=floor(size(X,1)./batch_size);
    xbatch = BatchSampler(X, batch_size);%产生批次数据
    gbatch= BatchSampler(grand, batch_size);
    
    iteration = 0;
    for epoch=1:Epoch%1000参考代码XXX
         %% 前向传播
         for j=1:batchnum
             iteration = iteration+1;
             x1=xbatch{j};
             grand = gbatch{j};
             x1_1 = dlarray(x1', 'CBT');
             pred1=forward(net,x1_1);
             pred = pred1.extractdata;
             
             % null = SAPredict(x1(1,:),sur);
             null = sur(x1(1,:));

             null=softmax(null);
             % 判断 null 张量的形状
             if numel(size(null)) == 1
                 % null 张量是 1 维，将其重新调整为 (1, 1) 形状
                 null = reshape(null, 1, 1);
             end
             %dacefit-S1

             % values=helpPre(x1,sur);
             values=sur(x1);
             
             penalty=mse(sum(pred,2),grand-null);
             S=ShapleySampler(num_players,batch_size*num_samples,0);
             S1 = reshape(S, [num_samples,num_players,batch_size]);
             pred1=reshape(pred,[num_players,1,batch_size]);
             
             approx = zeros(batch_size,num_samples, 1);
             
             for ii = 1:batch_size
                 st1= squeeze(S1(:, :, ii));
                 st2=squeeze(pred1(:, :, ii));
                 st3=st1*st2;
                 approx(ii, :, :) = st3;
             end
             t2=approx;
             
             X1 = dlarray(t2', 'CBT');
             T = dlarray(values', 'CBT');
             [loss, gradients] = dlfeval(@modelLossTripletSearcher, net, X1, T,penalty,lambda);
             gradients = dlupdate(@(g,w) g + l2Regularization*w, gradients, net.Learnables);
             [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, averageGrad, averageSqGrad, iteration, learnRate,gradientDecayFactor, squaredGradientDecayFactor);
         end
     end
    obj=net;
    
end
function [loss,gradients] = modelLossTripletSearcher(net,X,T,penalty,lambda)
lossc = mse(X,T);
loss = 1.0*lossc+penalty*lambda;
gradients = dlgradient(loss,net.Learnables);
end



