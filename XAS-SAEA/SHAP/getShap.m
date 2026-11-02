function SHAP=getShap(Model,Dec,Problem)
% 获取Shapley value
    SHAP=zeros(size(Dec,1),size(Dec,2));

    netF=FastShap(Problem,Model,Dec);
    %计算shap值
    Xtemp = dlarray(Dec', 'CBT');
    T1=forward(netF,Xtemp);
    T11=T1.extractdata;
    T2=double(T11');
    SHAP=T2;

end