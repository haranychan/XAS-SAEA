function grand = calculate_grand_coalition(x,Model, batch_size)
    % 计算每个x的总联盟价值。

    % 参数:
    %   x: 输入数据。
    %   imputer: 用于预测的模型。
    %   batch_size: 批处理大小。
    %   link: 链接函数，用于对imputer输出进行变换。

    % 返回:
    %   grand: 每个x的总联盟价值。
%     one1 = ones(batch_size, num_players, 'single');
%     Sur=Surrogate(sur,size(X,2));
    grand = [];
    for i = 1:ceil(size(x,1)/ batch_size)
        start = (i-1) * batch_size + 1;
        finish = min(start + batch_size - 1, size(x,1));
        batch_x = x(start:finish,:);
        %        batch_ones = ones(1:(finish-start+1), size(x,2));    %有点问题
               % batch_output=forward(Sur,batch_x);

        % batch_output=helpPre(batch_x,Model);
        batch_output=Model(batch_x);

        
        %        batch_output = sur(batch_x,batch_ones);
        batch_output = softmax(batch_output);
        grand = [grand; batch_output];
    end  
    if size(grand, 2) == 1
        grand = reshape(grand, [], 1);
    end
end
 function [Z, subsetWeights] = coalitionMatrix(M, numsubsets)
           
  %M=特征数
            if numsubsets ==2  % numsubsets is always greater than 1 (already validated in fit())
                % there are two subsets with infinite weights, they will be added to Z before solving the least squares
                % return early for this case of a budget of only two coalitions
                Z = [];
                subsetWeights = [];
                warning(message('stats:shapley:MaxNumSubsetsTooSmall'));
                return;
            end

            if M==2 % only two features, spell out Z
                switch numsubsets
                    case 3
                        Z = [true false];
                        subsetWeights = 1/2;
                    case 4
                        Z = [true false
                            false true];
                        subsetWeights = [1/2; 1/2];
                end
            else
                numsubsetsLargeEnough = false; % numsubsets is large enough to enumerate all subsets (2^M - 2 subsets)
                hasOddNumDimensions = mod(M,2)==1; % for odd number of features every single coalition has a complement e.g. (3choose0) + (3choose1) | (3choose2) + (3chose3)
                halfNumDimensions = floor(M/2); % symmetric problem
                featureSubsetSizes = uint64(1:halfNumDimensions)'; % a column of size halfNumDimensions
                binomialCoefficientsHalf= arrayfun(@(k)nchoosek(M,k), featureSubsetSizes); % a column of size halfNumDimensions
                INFINITY = intmax("uint64");
                cumulativeEnumerationsHalf = cumsum(binomialCoefficientsHalf); % same size as halfNumDimensions,
                % can contain overflown integers but that is ok, as long as the budget is reasonable, but if the specified budget also exceeds intmax, throw an error
                if any(cumulativeEnumerationsHalf) == INFINITY && budget > INFINITY
                    error(message('stats:shapley:InfiniteNumSubsets'));
                end
                % ideally we want to enumerate all the subsets,
                totalSubsetsWhole = 2^M; % this is the total number of subsets
                if numsubsets >= totalSubsetsWhole  % if the numsubsets exceeds the total number of enumerations to solve this problem, we can obtain an exact solution
                    numsubsetsLargeEnough = true;
                else % numsubsets is less than required for enumeration of all possible subsets
                    if numsubsets == 2*M + 2 % the case when subsets of size 1 can be enumerated
                        featureSubsetSizeExact = 1;
                    elseif numsubsets > 2*M + 2 % the case when more subsets than just size 1 can be enumerated
                        for ind = 1:halfNumDimensions
                            if numsubsets < 2*cumulativeEnumerationsHalf(ind) + 2
                                featureSubsetSizeExact = featureSubsetSizes(ind-1); % number of pairs of subsets we can fully enumerate within the numsubsets
                                break;
                            end
                        end
                    else % for numsubsets less than 2*M + 2, here not all subsets of size 1 can be enumerated
                        numSubsetsMinus2 = numsubsets-2;
                        Z = false(numSubsetsMinus2,M);
                        subsetWeights = zeros(numSubsetsMinus2,1);
                        if numSubsetsMinus2 <= M % numsubsets are in the range (2, M+2]
                            Z(1:numSubsetsMinus2+1:numSubsetsMinus2*numSubsetsMinus2) = true;
                            subsetWeights(1:numSubsetsMinus2) = 1/M;
                        else % numsubsets in the range (M+2, 2M+2)
                            Z(1:numSubsetsMinus2+1:end) = true;
                            remainingSubsets = numSubsetsMinus2 - M;
                            Z(M+1:M+remainingSubsets,:) = ~Z(1:remainingSubsets,:); % fill the remaining with as many complements as you can
                            subsetWeights = (1/M)*(ones(numSubsetsMinus2,1));
                        end
                        warning(message('stats:shapley:MaxNumSubsetsTooSmall'));
                        return;
                    end
                end

                kernelWeightsHalf = zeros(halfNumDimensions,1);
                binomialCoefficientsHalf = double(binomialCoefficientsHalf); % cast to double for weight calculation
                for ind = 1:halfNumDimensions % here ind is |z| and allPossibleSubsetSizesHalf is values of (M choose |z|) computed apriori, essentially make a hashtable of kernel weights
                    kernelWeightsHalf(ind)= (M-1)/(binomialCoefficientsHalf(ind)*ind*(M-ind)); % (M-1)/((M choose |z|)*(|z|)*(M-|z|), where is z is the subset, and |z| is the subset size
                end

                if numsubsetsLargeEnough % we can fully enumerate all the subsets if the numsubsets is large enough to accomodate the size
                    exactSize = 2^M-2; % there two subsets which have all features included and none of the features included, handle them separately
                   
                    Z = false(exactSize, M);
                    subsetWeights = zeros(exactSize,1);
                    for featureSubsetSize = 1:halfNumDimensions
                        hotIndices = nchoosek(1:M, featureSubsetSize); % hot indices are of size n-by-M where n = M-choose-featureSubsetSize
                        rowSize = binomialCoefficientsHalf(featureSubsetSize);
                        if featureSubsetSize ==1
                            rowOffset = 0; % avoid zero indexing into cumulativeEnumerations
                        else
                            rowOffset = cumulativeEnumerationsHalf(featureSubsetSize-1); % totalPossibleEnumerations are the cumulative sum so far
                        end

                        kw = kernelWeightsHalf(featureSubsetSize);
                        for row = 1:rowSize
                            Z(row+rowOffset, hotIndices(row,:)) = true; % flip the false bits in this subset
                            subsetWeights(row+rowOffset) = kw; % kernel weights are symmetric so fill all of them out
                        end
                    end
                   
                    if hasOddNumDimensions % for odd number of dimensions every subset has its complement
                        Z(exactSize/2+1:end,:) = ~Z(1:exactSize/2,:); % add the complements to Z
                        subsetWeights(exactSize/2+1:end) = subsetWeights(1:exactSize/2); % the weights are the same
                    else
                       
                        rowEndSymmetryIdx = cumulativeEnumerationsHalf(end-1);
                        rowBeginSymmetryAgainIdx = cumulativeEnumerationsHalf(end)+1;
                        Z(rowBeginSymmetryAgainIdx:end,:) = ~Z(1:rowEndSymmetryIdx,:); % add the complements to Z
                        subsetWeights(rowBeginSymmetryAgainIdx:end) = subsetWeights(1:rowEndSymmetryIdx); % add the complements to Z
                    end

                else
                  
                    numSubsetsMinus2 = numsubsets-2; % exclude the two trivial subsets (taking all features or taking none), will be added later by fitKernelSHAP
                    Z = false(numSubsetsMinus2, M);
                    subsetWeights = zeros(numSubsetsMinus2,1);

                    for featureSubsetSize = 1:featureSubsetSizeExact
                        hotIndices = nchoosek(1:M, featureSubsetSize); % hot indices are of size n-by-M where n = M-choose-featureSubsetSize
                        rowSize = binomialCoefficientsHalf(featureSubsetSize);
                        if featureSubsetSize ==1
                            rowOffset = 0; % avoid zero indexing into cumulativeEnu
                        else
                            rowOffset = cumulativeEnumerationsHalf(featureSubsetSize-1); % totalPossibleEnumerations are the cumulative sum so far
                        end
                        kw = kernelWeightsHalf(featureSubsetSize);
                        for row = 1:rowSize
                            Z(row+rowOffset, hotIndices(row,:)) = true; % flip the false bits in this subset
                            subsetWeights(row+rowOffset) = kw; % kernel weights are symmetric so fill all of them out
                        end
                    end

                    % now do partial enumeration for the next subset in line that would have been fully enumerated if we had more numsubsets
                    halfBudget = ceil(numSubsetsMinus2/2);
                    rowEndSymmetryIdx = floor(numSubsetsMinus2/2); % take complements of subsets upto here
                    rowBeginSymmetryAgainIdx = halfBudget+1; % populate complements starting at this index
                    partialEnumerationBlock = row+rowOffset+1:halfBudget;
                    hotIndices = shapley.partiallyEnumeratedSubsets(1:M,featureSubsetSizeExact+1,numel(partialEnumerationBlock)); % this is the subset next in line to enumerate

                    for row = 1:numel(partialEnumerationBlock)
                        Z(partialEnumerationBlock(row), hotIndices(row,:)) = true;
                        subsetWeights(partialEnumerationBlock(row)) = kernelWeightsHalf(featureSubsetSizeExact+1);
                    end

                    Z(rowBeginSymmetryAgainIdx:end,:) = ~Z(1:rowEndSymmetryIdx,:); % add the complements to Z
                    subsetWeights(rowBeginSymmetryAgainIdx:end) = subsetWeights(1:rowEndSymmetryIdx); % add the complements to Z
                end
            end
  end
