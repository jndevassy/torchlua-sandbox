modelcifar10 = {}

modelcifar10.buildConvNet = function(inputFeatureMaps,nRowsOrCols,filterKernels,filterSize,poolSize,mlpHiddenUnits,outputUnits)
    local newSize = nRowsOrCols
    local model = nn.Sequential()
    -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
    model:add(nn.SpatialConvolutionMM(inputFeatureMaps, filterKernels[1], filterSize, filterSize))
    newSize = newSize-(filterSize-1) --reduced size of new feature maps; count = filterKernels[1]
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(poolSize,poolSize,poolSize,poolSize))
    newSize = newSize/2 --reduced size of new feature maps
    -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
    model:add(nn.SpatialConvolutionMM(filterKernels[1], filterKernels[2], filterSize, filterSize))
    newSize = newSize-(filterSize-1) --reduced size of new feature maps; count = filterKernels[2]
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(poolSize,poolSize,poolSize,poolSize))
    newSize = newSize/2 --reduced size of new feature maps
    -- stage 3 : standard 2-layer neural network
    --new feature maps are of size newSize x newSize and there are filterKernels[2] of these; flatten them out
    --this is the input layer for the fully connected mlp
    model:add(nn.Reshape(filterKernels[2]*newSize*newSize))
    model:add(nn.Dropout(0.5))
    --middle layer to have mlpHiddenUnits number of units
    model:add(nn.Linear(filterKernels[2]*newSize*newSize, mlpHiddenUnits))
    model:add(nn.ReLU())
    --output layer to have outputUnits units
    model:add(nn.Linear(mlpHiddenUnits, outputUnits))
    model:cuda()
    local crit = nn.CrossEntropyCriterion()
    crit:cuda()
    return model,crit
end

modelcifar10.train = function (dataSet,model,criterion,options,confusionMatrix,logger)
    -- local vars
    local parameters,df_dW = model:getParameters()
    local time = sys.clock()
    -- set model to training mode (for modules that differ in training and testing, like Dropout)
    model:training()
    -- shuffle at each epoch
    local shuffle = torch.randperm(dataSet.data:size(1))
    -- do one epoch
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. options.trainingEpoch .. ' [batchSize = ' .. options.batchSize .. ']')
    for t = 1,dataSet.data:size(1),options.batchSize do
        -- disp progress
        xlua.progress(t, dataSet.data:size(1))
        -- create mini batch
        local inputs = {}
        local targets = {}
        for i = t,math.min(t+options.batchSize-1,dataSet.data:size(1)) do
            -- load new sample
            local input = dataSet.data[shuffle[i]]
            local target = dataSet.label[shuffle[i]]
            table.insert(inputs, input)
            table.insert(targets, target)
        end
        -- create closure to evaluate f and df/dW
        local feval = function()
            -- reset gradients
            df_dW:zero()
            -- f is the average of all criterions
            local f = 0
            -- evaluate function for complete mini batch
            for i = 1,#inputs do
                -- estimate f
                local output = model:forward(inputs[i])
                local err = criterion:forward(output, targets[i])
                f = f + err
                -- estimate df/dW
                local df_do = criterion:backward(output, targets[i])
                --weighted delta from next layer brought back times gradient of previous layer activation:
                local gradWrtInput = model:updateGradInput(inputs[i],df_do)
                --gradWrtInput at next layer times the previous layer activation:
                model:accGradParameters(inputs[i],df_do)
                -- update confusion
                confusionMatrix:add(output, targets[i])
            end
            -- normalize gradients and f
            df_dW:div(#inputs)
            f = f/#inputs
        end
        -- update parameters using df_dW as per
        -- parameters = parameters - learningRate * df_dW
        feval()
        model:updateParameters(options.learningRate)
    end
    -- time taken
    time = sys.clock() - time
    time = time / dataSet.data:size(1)
    print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
    -- print confusion matrix
    print(confusionMatrix)
    -- update logger/plot
    --totalValid is the sum of the diagonal of the confusion matrix divided by the sum of the matrix. 
    --averageValid is the average of all diagonals divided by their respective rows.
    logger:add{['% mean class accuracy (train set)'] = confusionMatrix.totalValid * 100}
    logger:style{['% mean class accuracy (train set)'] = '-'}
    logger:plot()
    -- save/log current net
    local filename = paths.concat(options.savePath, 'model.net')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    print('==> saving model to '..filename)
    torch.save(filename, model)
    -- next epoch
    confusionMatrix:zero()
    options.trainingEpoch = options.trainingEpoch + 1
end

return modelcifar10

