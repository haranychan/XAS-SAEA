function explainer=Explainer(Problem)
% 根据当前问题维度，创建模型
    layers = [sequenceInputLayer(Problem.D, 'Name', 'seq_1')
              fullyConnectedLayer(Problem.D, 'Name', 'fcl_1')
              reluLayer('Name', 'relu_1')
              fullyConnectedLayer(Problem.D, 'Name', 'fcl_2') %%%输出anchor middle solution 
              reluLayer('Name', 'relu_2')
              fullyConnectedLayer(Problem.D, 'Name', 'fcl_3')  %%%输出anchor middle solution
               reluLayer('Name', 'relu_3')
               fullyConnectedLayer(	Problem.D, 'Name', 'fcl_4') %%%输出anchor middle solution 
%               reluLayer('Name', 'relu_4')
    ];
    lgraph = layerGraph(layers);
    net = dlnetwork(lgraph);    
   
    explainer=net;
    
end
function [loss,gradients] = modelLossTripletSearcher(net,X,T1)
    lossc = mse(X,T1);
    %     loss = lossa + 0.5*lossc;
    loss =1.0*lossc;
    gradients = dlgradient(loss,net.Learnables);
end
