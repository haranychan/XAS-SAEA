function [train_x, train_y, train_p] = GetTrain_XASSAEA(id, db_x, db_y, pred_his, pop, N, change_data_n, ch_d_n_thres, data_n, data_n_2, data_times)
    D = size(db_x, 2);
    switch id
        case 1  % All Data
            train_x = db_x; train_y = db_y; train_p = pred_his;
        case 2  % Top
            [B, I] = sort(db_y);
            if change_data_n == 1 && D >= ch_d_n_thres
                train_x = db_x(I(1:min(data_n_2, size(db_x, 1))), :); train_y = B(1:min(data_n_2, size(db_x, 1)));
                train_p = pred_his(I(1:min(data_n_2, size(db_x, 1))));
            elseif change_data_n == 2
                train_x = db_x(I(1:min(data_times*D, size(db_x, 1))), :); train_y = B(1:min(data_times*D, size(db_x, 1)));
                train_p = pred_his(I(1:min(data_times*D, size(db_x, 1))));
            else
                train_x = db_x(I(1:N), :); train_y = B(1:N); train_p = pred_his(I(1:N));
            end
        case 3  % Recent
            train_x = db_x(end-N+1:end, :); train_y = db_y(end-N+1:end); train_p = pred_his(end-N+1:end);
        case 4  % Neighbor
            dists = pdist2(pop, db_x); [~, I] = sort(dists, 2); I = unique(I(:, 1:min(data_n, size(db_x, 1))));
            train_x = db_x(I, :); train_y = db_y(I); train_p = pred_his(I);
        case 5  % Near Best
            [~, I] = sort(db_y); b1 = db_x(I(1), :); dist = pdist2(b1, db_x); [~, I] = sort(dist);
            if change_data_n == 1 && D >= ch_d_n_thres
                train_x = db_x(I(1:min(data_n_2, size(db_x, 1))), :); train_y = db_y(I(1:min(data_n_2, size(db_x, 1))));
                train_p = pred_his(I(1:min(data_n_2, size(db_x, 1))));
            elseif change_data_n == 2
                train_x = db_x(I(1:min(data_times*D, size(db_x, 1))), :); train_y = db_y(I(1:min(data_times*D, size(db_x, 1))));
                train_p = pred_his(I(1:min(data_times*D, size(db_x, 1))));
            else
                train_x = db_x(I(1:N), :); train_y = db_y(I(1:N)); train_p = pred_his(I(1:N));
            end
    end
end

