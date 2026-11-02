function S=ShapleySampler(num_players,batch_size,paired_sampling)
  
            arange = [1:num_players-1];
            w = 1 ./ (arange .* (num_players - arange));
            w = w / sum(w);
%             categorical1 = categorical(w);
            categorical1 = makedist('Multinomial','probabilities',w);
            %             obj.num_players = num_players;
            tril1 = tril(ones(num_players - 1, num_players, 'single'), 0);

            s1=size(categorical1.Probabilities,2);
            samples = randsample(s1, batch_size, true, categorical1.Probabilities);

            num_included = 1 + samples;
            S = tril1(num_included - 1, :);
            
            for i = 1:batch_size
                if paired_sampling==1 && mod(i, 2) == 1
                    S(i, :) = 1 - S(i - 1, :);
                else
                    ind=randperm(num_players,num_players);
                     S(i,:)=S(i,ind);
                end
            end
        end
   