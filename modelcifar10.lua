modelcifar10 = {}

modelcifar10.buildCriterion = function ()
    local crit = nn.ClassNLLCriterion()
    crit:cuda()
    return crit
end

modelcifar10.buildConvNet = function(inputFeatureMaps,nRowsOrCols,filterKernels,filterSize,poolSize,mlpHiddenUnits,outputUnits)
    local newSize = nRowsOrCols
    local normkernel = image.gaussian1D(7)
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
    model:add(nn.View(filterKernels[2]*newSize*newSize))
    model:add(nn.Dropout(0.5))
    --1st mlp hidden layer to have mlpHiddenUnits[1] number of units
    model:add(nn.Linear(filterKernels[2]*newSize*newSize, mlpHiddenUnits[1]))
    model:add(nn.Tanh())
    --2nd mlp hidden layer to have mlpHiddenUnits[2] number of units
    model:add(nn.Linear(mlpHiddenUnits[1], mlpHiddenUnits[2]))
    model:add(nn.Tanh())
    --output layer to have outputUnits units
    model:add(nn.Linear(mlpHiddenUnits[2], outputUnits))
    --convert output to log-probabilities
    model:add(nn.LogSoftMax())
    model:cuda()
    local crit = modelcifar10.buildCriterion()
    return model,crit
end

modelcifar10.train = function (dataSet,model,criterion,options,confusionMatrix,logger,sgdOptimizeCall,optimizerState,modelFilename)
    local w,dE_dw = model:getParameters()
    local time = sys.clock()
    -- set model to training mode (for modules that differ in training and testing, like Dropout)
    model:training()
    confusionMatrix:zero()
    options.trainingEpoch = options.trainingEpoch + 1
    -- shuffle at each epoch
    local shuffle = torch.randperm(dataSet.data:size(1))
    -- do one epoch
    print('==> TRAINING EPOCH: ---')
    print("==> online# " .. options.trainingEpoch .. ' [batchSize = ' .. options.batchSize .. ']')
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
                -- the output of the network is log-probabilities. take e^x to get actual probability values
                output:exp()
                confusionMatrix:add(output, targets[i])
            end
            -- normalize gradients and cost E
            dE_dw:div(#inputs)
            E = E/#inputs
            -- return cost and gradients for mini batch
            return E,dE_dw
        end
        -- perform gradient descent for mini batch just created
        if not options.useOptimizer then
            gradientDescent(w)
            -- update parameters/weights using dE_dw as per:
            -- w = w - learningRate * dE_dw
            model:updateParameters(options.learningRate)
        else
            sgdOptimizeCall(gradientDescent,w,optimizerState)
        end
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
    logger:add{["1"] = confusionMatrix.totalValid * 100}
    -- save/log current net
    local filename = paths.concat(options.savePath, modelFilename)
    os.execute('mkdir -p ' .. sys.dirname(filename))
    print('==> saving model to '..filename)
    torch.save(filename, model)
end

modelcifar10.test = function (dataSet,model,confusionMatrix,logger)
    local time = sys.clock()
    -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
    model:evaluate()
    confusionMatrix:zero()
    -- test over test data
    print('==> TESTING: ---')
    for t = 1,dataSet.data:size(1) do
        -- disp progress
        xlua.progress(t, dataSet.data:size(1))
        -- get new sample
        local input = dataSet.data[t]
        local target = dataSet.label[t]
        -- test sample
        local predicted = model:forward(input)
        -- the output of the network is log-probabilities. take e^x to get actual probability values
        predicted:exp()
        confusionMatrix:add(predicted, target)
    end
    -- timing
    time = sys.clock() - time
    time = time / dataSet.data:size(1)
    print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')
    -- print confusion matrix
    print(confusionMatrix)
    -- update log/plot
    logger:add{["1"] = confusionMatrix.totalValid * 100}
end

modelcifar10.testSingleImage = function (dataSet,model,classes,sampleNum)
    -- get sample
    local input = dataSet.data[sampleNum]
    gfx.image(input)
    local target = dataSet.label[sampleNum]
    -- test sample
    local predicted = model:forward(input)
    -- the output of the network is Log-Probabilities. To convert them to probabilities, you have to take e^x 
    local logp = torch.exp(predicted)
    print('actual: '..classes[target])
    for i=1,predicted:size(1) do
        print(classes[i], predicted[i], logp[i])
    end
end

return modelcifar10

