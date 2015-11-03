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
    local w,dE_dw = model:getParameters()
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
            -- load new sample and target
            local input = dataSet.data[shuffle[i]]
            local target = dataSet.label[shuffle[i]]
            table.insert(inputs, input)
            table.insert(targets, target)
        end
        -- create closure to evaluate cost E and dE/dw
        local gradientDescent = function(weights)
            -- overwrite the weights if they are being reset by client; w <= weights
            if w ~= weights then
                w:copy(weights)
            end
            -- reset gradients
            dE_dw:zero()
            -- E is the average of all criterions (cost)
            local E = 0
            -- evaluate cost and gradients for complete mini batch
            for i = 1,#inputs do
                -- estimate cost E
                local output = model:forward(inputs[i])
                local err = criterion:forward(output, targets[i])
                E = E + err
                -- estimate dE_dw
                local dE_do = criterion:backward(output, targets[i])
                --gradient wrt input: weighted delta from next layer times gradient of previous layer activation
                model:updateGradInput(inputs[i],dE_do)
                --gradient wrt parameters/weights: gradient wrt input at next layer times the previous layer activation
                model:accGradParameters(inputs[i],dE_do)
                -- update confusion
                confusionMatrix:add(output, targets[i])
            end
            -- normalize gradients and cost E
            dE_dw:div(#inputs)
            E = E/#inputs
        end
        -- perform gradient descent for mini batch just created
        gradientDescent(w)
        -- update parameters/weights using dE_dw as per:
        -- w = w - learningRate * dE_dw
        model:updateParameters(options.learningRate)
    end
    -- time taken for complete dataSet
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
    -- prepare for next epoch
    confusionMatrix:zero()
    options.trainingEpoch = options.trainingEpoch + 1
end

modelcifar10.test = function (dataSet,model,confusionMatrix,logger)
    local time = sys.clock()
    -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
    model:evaluate()
    -- test over test data
    print('==> testing on test set:')
    for t = 1,dataSet.data:size(1) do
        -- disp progress
        xlua.progress(t, dataSet.data:size(1))
        -- get new sample
        local input = dataSet.data[t]
        local target = dataSet.label[t]
        -- test sample
        local pred = model:forward(input)
        confusionMatrix:add(pred, target)
    end
    -- timing
    time = sys.clock() - time
    time = time / dataSet.data:size(1)
    print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')
    -- print confusion matrix
    print(confusionMatrix)
    -- update log/plot
    logger:add{['% mean class accuracy (test set)'] = confusionMatrix.totalValid * 100}
    logger:style{['% mean class accuracy (test set)'] = '-'}
    logger:plot()
    -- next iteration:
    confusionMatrix:zero()
end

return modelcifar10

