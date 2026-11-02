function y = softmax(x)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

%     c = max(a);
%     exp_a = exp(a -c);
%     sum_exp_a = sum(exp_a);
%
%     y = exp_a ./ sum_exp_a;



    if ismatrix(x)

        x = x';
        x = x - max(x);
        y = exp(x) ./ sum(exp(x));

        y = y';

    else

        x = x - max(x); % to avoid overflow
        y = exp(x) ./ sum(exp(x));

    end

end

