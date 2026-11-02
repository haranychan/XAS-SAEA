classdef XASSAEA < ALGORITHM
% <2026> <single> <real/integer> <expensive>
% eXplainable Adaptive Switching-based Surrogate-assisted Evolutionary Algorithm
% N             --- 100   --- Population size of DE
% F             --- 0.5   --- Scaling factor of DE
% CR            --- 0.9   --- Crossover rate of DE
% mut           --- 3     --- Mutation strategy ID 
% xov           --- 1     --- Crossover strategy ID
% datacrtr_id   --- 5     --- {1, 2, 3, 4, 5} : {All, Top, Recent, Neighbor, Near Best}
% change_data_n --- 2     --- {0, 1, 2} : whether change datasize of second stage {0: False, 1:True-fix, 2:True-nD}
% ch_d_n_thres  --- 100   --- Threshold to change datasize if change_data_n = 1
% data_n        --- 100   --- The number of neighbor if change_data_n = 0
% data_n_2      --- 200   --- The number of data size if change_data_n = 1
% data_times    --- 5     --- Parameter to define data size if change_data_n = 2
% shap_nbr_n    --- 30    --- The number of neighbor solutions for SHAP
% acc_window    --- 20    --- The number of points to calculate the accuracy
% tau_u_rate    --- 0.8   --- Upper bound rate for tau
% tau_l_rate    --- 0.5   --- Lower bound rate for tau
% perturb_times --- 0.01  --- Parameter to define the perturbation range

%------------------------------- Reference --------------------------------
% K. Nishihara, T. Sato, K. Izui, and S. Watanabe. Adaptive Switching 
% between Search and Model Refinement via Explainable Machine Learning in 
% Surrogate-assisted Evolutionary Algorithms, Proceedings of the IEEE 
% Congress on Evolutionary Computation, 2026, cec_pap649.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kei Nishihara (email: nishihara-kei-pv@ynu.ac.jp)

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [N,F,CR,mut,xov,datacrtr_id,change_data_n,ch_d_n_thres,data_n,data_n_2,data_times,shap_nbr_n,acc_window,tau_u_rate,tau_l_rate,perturb_times,kernel] ...
                = Algorithm.ParameterSet(100,0.5,0.9,3,1,5,2,100,100,200,5,30,20,0.8,0.5,0.01,'CB');
            
            D = Problem.D;

            %% Generate random population
            PopDec         = UniformPoint(N, D, 'Latin');
            Population     = Problem.Evaluation(repmat(Problem.upper - Problem.lower, N, 1) .* PopDec + repmat(Problem.lower, N, 1));
            Arc            = Population;

            % Archive
            db_x = Population.decs; db_y = Population.objs;

            pred_his = NaN(N, 1); acc_normalized_his = NaN(N, 1); delta_shap_his = NaN(N, 1);
            is_OptimizationMode = 0; % {0, 1} : {Model-refinement mode, Optimization mode}
            
            gen = 1;
            
            %% Optimization
            while Algorithm.NotTerminated(Arc)
                
                %% Obtain parent solutions
                [B, I] = sort(db_y); 
                pop = db_x(I(1:N), :); fit = B(1:N);
                
                %% Obtain training data and Construct surrogate model
                [train_x, train_y, train_p] = GetTrain_XASSAEA(datacrtr_id, db_x, db_y, pred_his, pop, N, change_data_n, ch_d_n_thres, data_n, data_n_2, data_times);
                surr = struct; surr.train_x = train_x;
                [surr.lambda, surr.gamma, surr.rho] = RBF(train_x, train_y, kernel);
                surr = @(x) RBF_eval(x, surr.train_x, surr.lambda, surr.gamma, surr.rho, kernel);
                
                %% Calculate metrics
                std_y = std(train_y(~isnan(train_p)));
                acc = mean(abs(train_p - train_y), 'omitnan');
                acc_normalized = acc / (std_y + 1e-10);
                acc_normalized_his = [acc_normalized_his; acc_normalized];

                shap = getShap(surr, train_x(1:min(shap_nbr_n, size(train_x, 1)), :), Problem);
                shap_abs  = abs(shap);
                shap_abs_unit = shap_abs ./ sum(shap_abs, 2);
                [~, train_bid] = min(train_y(1:min(shap_nbr_n, size(train_x, 1))));
                shap_center = shap_abs_unit(train_bid, :);
                shap_abs_unit(train_bid, :) = [];
                delta_shap = mean(sum(abs(shap_abs_unit - shap_center), 2, 'omitnan'), 1);
                delta_shap_his = [delta_shap_his; delta_shap];

                %% Mode switch
                if Problem.FE < N + acc_window
                    is_OptimizationMode = 0;
                else
                    tau_acc = prctile(acc_normalized_his(end - acc_window : end - 1), [tau_l_rate, tau_u_rate]*100);
                    tau_delta_shap = prctile(delta_shap_his(end - acc_window : end - 1), [tau_l_rate, tau_u_rate]*100);
                    if acc_normalized > tau_acc(2) || delta_shap > tau_delta_shap(2)        % inaccurate or messy-structure
                        is_OptimizationMode = 0;
                    % between tau_*(1) and tau_*(2) : hysteresis (margin)
                    elseif acc_normalized < tau_acc(1) && delta_shap < tau_delta_shap(1)    % accurate and ordered-structure
                        is_OptimizationMode = 1;
                    end
                end

                %% Prescreening
                ub = max(train_x, [], 1); lb = min(train_x, [], 1);
                if is_OptimizationMode
                    cand = DEGetTrialVector(pop, fit, lb, ub, F, CR, mut, xov);
                    pred_Fs = surr(cand);
                    [pred_F, idx] = min(pred_Fs);
                    pred_his = [pred_his; pred_F];
                    offspringDec = cand(idx, :);
                else
                    center_x = train_x(train_bid, :);
                    weight = mean(abs(shap_abs_unit - shap_center), 1) ./ delta_shap * D;
                    if anynan(weight)
                        offspringDec = center_x + perturb_times .* (Problem.upper - Problem.lower) .* randn(1, D);
                    else
                        offspringDec = center_x + perturb_times .* (Problem.upper - Problem.lower) .* weight .* randn(1, D);
                    end
                    offspringDec = max(min(offspringDec, Problem.upper), Problem.lower);
                    pred_F = surr(offspringDec);
                    pred_his = [pred_his; pred_F];
                end
               
                %% Evaluation
                offspring = Problem.Evaluation(offspringDec);
                Arc       = [Arc, offspring];
                Algorithm.NotTerminated(Arc);

                db_x = [db_x; offspring.dec];  db_y = [db_y; offspring.obj];
                [db_x, ia, ~] = unique(db_x, 'stable', 'rows'); db_y = db_y(ia);
                
                gen = gen + 1;
            end
        end
    end
end