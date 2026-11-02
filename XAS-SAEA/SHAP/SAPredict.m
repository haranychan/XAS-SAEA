function val=SAPredict(Next,Smodel)
%代理模型前向传播/predict过程
    model_x = Smodel.X;    
    C1_data = model_x(Smodel.Y ==1,:);
    C2_data = model_x(Smodel.Y ~=1,:);

    C1_num   = size(C1_data,1);
    C2_num   = size(C2_data,1);
    Next_num = size(Next,1);

    scores = zeros(Next_num,2);
    
    all_testdata = zeros(2*(C1_num+C2_num)*Next_num,2*size(C1_data,2));
    for i = 1 : size(Next,1)
        original = (i-1)*2*(C1_num+C2_num);
        Xi       = repmat(Next(i,:),size(C1_data,1),1);
        all_testdata(original+1:original+C1_num,:)          = [C1_data,Xi];  %C1_Xi
        all_testdata(original+1+C1_num:original+C1_num*2,:) = [Xi,C1_data]; %Xi_C1
        
        Xi = repmat(Next(i,:),size(C2_data,1),1);
        all_testdata(original+1+C1_num*2:original+C1_num*2+C2_num,:)          = [C2_data,Xi]; %C2_Xi
        all_testdata(original+1+C2_num+C1_num*2:original+C2_num*2+C1_num*2,:) = [Xi,C2_data];%Xi_C2
    end
    
    TestIn_nor = mapminmax('apply',all_testdata',Smodel.mp_struct)';
    pre_out = Smodel.net(TestIn_nor')';  
    
    for i = 1 : size(Next,1)
        C_SCORE    = zeros(1,2);
        original   = (i-1)*2*(C1_num+C2_num);
        pre_C1Xi   = sum(pre_out(original+1:original+C1_num,:),1)./C1_num;
        C_SCORE(1) = C_SCORE(1) + pre_C1Xi(2)+pre_C1Xi(3);   
        C_SCORE(2) = C_SCORE(2) + pre_C1Xi(1);               
        
        pre_XiC1   = sum(pre_out(original+1+C1_num:original+C1_num*2,:),1)./C1_num;
        C_SCORE(1) = C_SCORE(1) + pre_XiC1(2) + pre_XiC1(1);  
        C_SCORE(2) = C_SCORE(2) + pre_XiC1(3);                 
        
        pre_C2Xi   = sum(pre_out(original+1+C1_num*2:original+C1_num*2+C2_num,:),1)./C2_num;
        C_SCORE(1) = C_SCORE(1) + pre_C2Xi(3);
        C_SCORE(2) = C_SCORE(2) + pre_C2Xi(2) + pre_C2Xi(1);
        
        pre_XiC2   = sum(pre_out(original+1+C2_num+C1_num*2:original+C2_num*2+C1_num*2,:),1)./C2_num;
        C_SCORE(1) = C_SCORE(1) + pre_XiC2(1);
        C_SCORE(2) = C_SCORE(2) + pre_XiC2(2) + pre_XiC2(3);
        
        scores(i) = C_SCORE(1)-C_SCORE(2);
    end      
    [~,ind] = sort(scores,'descend'); 
    val=scores(:,1);
end