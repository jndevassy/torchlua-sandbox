require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'xlua'
require 'image'
require 'gfx.js'

--[[command line arguments, example '$> th nntest.lua --batchSize 128 --momentum 0.5' ]]--
cmd = torch.CmdLine()
cmd:option('-save', '/home/mit/projects/thtests/results', 'subdirectory to save/log experiments in')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
opt = cmd:parse(arg or {})
print('processing options ==>',opt)
gfx.startserver()

--GLOBALS
train_file = 'train_32x32.t7'
test_file = 'test_32x32.t7'
trainSize = 10000
testSize = 2000
--data
trainData = trainData or {}
testData = testData or {}
channels = {'y','u','v'}
-- input channels or feature maps
nfeaturemaps = 3
nRowsOrCols = 32 --image is 32 x 32
-- filter sizes
nfilterKernelsByLayer = {64,64}
nfiltsize = 5
npoolsize = 2
--mlp hidden units
nMLPHiddenUnits = 128
-- 10-class problem
noutputs = 10
-- classes
classes = {'1','2','3','4','5','6','7','8','9','0'}
-- optimizer settings ==>
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = 1e-7
}
optimMethod = optim.sgd
-- This matrix records the current confusion across classes
-- confusion:add(predicted,label) ==> confusion[label][predicted] = confusion[label][predicted] + 1
-- target label is "printed" on rows, predicted counts are shown on columns
-- ideally only diagonal elements should get updated i.e target label==predicted output
confusion = optim.ConfusionMatrix(classes)
-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
-- net and training vars
epoch,shuffle = nil,nil
nnet,criterion = nil,nil

loadSVHNData = function ()
    -- We load the dataset from disk, and re-arrange it to be compatible
    -- with Torch's representation. Matlab uses a column-major representation,
    -- Torch is row-major, so we just have to transpose the data.
    -- Note: the data, in X, is 4-d: the 1st dim indexes the samples, the 2nd
    -- dim indexes the color channels (RGB), and the last two dims index the
    -- height and width of the samples.
    print '==> downloading dataset'
    local www = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/housenumbers/'
    if not paths.filep(train_file) then
       os.execute('wget ' .. www .. train_file)
    end
    if not paths.filep(test_file) then
       os.execute('wget ' .. www .. test_file)
    end
    --training data --NCWH ==> NCHW transpose 3rd and 4th
    local loaded = torch.load(train_file,'ascii')
    trainData = {
       data = loaded.X:transpose(3,4),
       labels = loaded.y[1],
       size = function() return trainSize end
    }
    --test data --NCWH ==> NCHW transpose 3rd and 4th
    loaded = torch.load(test_file,'ascii')
    testData = {
       data = loaded.X:transpose(3,4),
       labels = loaded.y[1],
       size = function() return testSize end
    }
end

preprocessData = function ()
    trainData.data = trainData.data:float()
    testData.data = testData.data:float()
    -- For natural images, we use several intuitive tricks:
    --   + images are mapped into YUV space, to separate luminance information
    --     from color information
    --   + color channels are normalized globally, across the entire dataset;
    --     as a result, each color component has 0-mean and 1-norm across the dataset.  
    --   + the luminance channel (Y) is locally normalized, using a contrastive
    --     normalization operator: for each neighborhood, defined by a Gaussian
    --     kernel, the mean is suppressed, and the standard deviation is normalized
    --     to one.
    -- Convert all images to YUV
    print '==> preprocessing data: colorspace RGB -> YUV'
    for i = 1,trainData:size() do
        trainData.data[i] = image.rgb2yuv(trainData.data[i])
    end
    for i = 1,testData:size() do
        testData.data[i] = image.rgb2yuv(testData.data[i])
    end  
    -- GLOBAL NORMALIZATION ==>
    -- Normalize each channel, and store mean/std
    -- per channel. These values are important, as they are part of
    -- the trainable parameters. At test time, test data will be normalized
    -- using these values.
    print '==> preprocessing data: normalize each feature (channel) globally'
    local mean = {}
    local std = {}
    for i,channel in ipairs(channels) do
        -- normalize each channel globally across samples:
        mean[i] = trainData.data[{ {},i,{},{} }]:mean()
        std[i] = trainData.data[{ {},i,{},{} }]:std()
        print (channel..'mean='..mean[i])
        print (channel..'std='..std[i])
        trainData.data[{ {},i,{},{} }]:add(-mean[i])
        trainData.data[{ {},i,{},{} }]:div(std[i])
    end
    -- Normalize test data, using the training means/stds
    for i,channel in ipairs(channels) do
        -- normalize each channel globally across samples:
        testData.data[{ {},i,{},{} }]:add(-mean[i])
        testData.data[{ {},i,{},{} }]:div(std[i])
    end
    -- Local normalization ==>
    print '==> preprocessing data: normalize all three channels locally'
    -- Define the normalization neighborhood:
    local neighborhood = image.gaussian1D(13)
    -- Define our local normalization operator (It is an actual nn module, 
    -- which could be inserted into a trainable model):
    -- SpatialContrastiveNormalization(nInputPlane, kernel, threshold, thresval)
    local normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()
    -- Normalize all channels locally per sample:
    for c in ipairs(channels) do
       for i = 1,trainData:size() do
            trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }])
       end
       for i = 1,testData:size() do
            testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
       end
    end
end

verifyStats = function ()
    for i,channel in ipairs(channels) do
        local trainMean = trainData.data[{ {},i }]:mean()
        local trainStd = trainData.data[{ {},i }]:std()
        local testMean = testData.data[{ {},i }]:mean()
        local testStd = testData.data[{ {},i }]:std()
        print('training data, '..channel..'-channel, mean: ' .. trainMean)
        print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)
        print('test data, '..channel..'-channel, mean: ' .. testMean)
        print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
    end
end

visualizeSamples = function (numSamples)
    local samples_y = trainData.data[{ {1,numSamples},1 }]
    local samples_u = trainData.data[{ {1,numSamples},2 }]
    local samples_v = trainData.data[{ {1,numSamples},3 }]
    gfx.image(samples_y)
    gfx.image(samples_u)
    gfx.image(samples_v)
end

buildConvNet = function (inputFeatureMaps,nRowsOrCols,filterKernels,filterSize,poolSize,hiddenUnits,outputUnits)
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
    --middle layer to have hiddenUnits number of units
    model:add(nn.Linear(filterKernels[2]*newSize*newSize, hiddenUnits))
    model:add(nn.ReLU())
    --output layer to have outputUnits units
    model:add(nn.Linear(hiddenUnits, outputUnits))
    model:cuda()
    local crit = nn.CrossEntropyCriterion()
    crit:cuda()
    return model,crit
end

buildLeNet = function (inputChannels,inputSize)
    local newSize = inputSize
    local net = nn.Sequential()
    net:add(nn.SpatialConvolution(inputChannels,6,5,5))  --32x32 => 28x28; 32-(5-1) x 32-(5-1)
    newSize = newSize-(5-1)
    net:add(nn.SpatialMaxPooling(2,2,2,2))   --28x28 => 14x14; x 1/2
    newSize = newSize/2
    net:add(nn.SpatialConvolution(6,16,5,5)) --14x14 => 10x10; 14-(5-1) x 14-(5-1)
    newSize = newSize-(5-1)
    net:add(nn.SpatialMaxPooling(2,2,2,2))   --10x10 => 5x5  ; x 1/2
    newSize = newSize/2
    net:add(nn.Reshape(16*newSize*newSize))     --1 pixel as 1 neuron (400 total)
    net:add(nn.Linear(16*newSize*newSize,128))  --add hidden layer with 128 neurons
    net:add(nn.Linear(128,64))               --add another hidden layer
    net:add(nn.Linear(64,10))                --add output layer 10 digits to classify
    net:cuda()
    local crit = nn.CrossEntropyCriterion()
    crit:cuda()
    return net,crit
end

simplePlot = function ()
    if xlua.require('gnuplot') then
        symbols={}
        symbols[1]={}
        for i = 1,100 do
            table.insert(symbols[1],2*i -10)
        end
        symbols[2]={}
        for i = 1,100 do
            table.insert(symbols[2],100*math.log10(i))
        end
        plots={}
        for name,list in pairs(symbols) do
            plotlist = torch.Tensor(#list)
            for j = 1,#list do plotlist[j] = list[j] end
            table.insert(plots,{tostring(name),plotlist,'-'})
        end
        gnuplot.plot(plots)
    end
end

trainModel = function (theModel,crit)
    --   + construct mini-batches on the fly
    --   + define a closure to estimate (a noisy) loss
    --     function, as well as its derivatives wrt the parameters of the
    --     model to be trained
    --   + optimize the function, according to several optmization
    --     methods: SGD, L-BFGS.
    -- epoch tracker
    epoch = epoch or 1
    -- local vars
    local parameters,gradParameters = theModel:getParameters()
    local time = sys.clock()
    -- set model to training mode (for modules that differ in training and testing, like Dropout)
    theModel:training()
    -- shuffle at each epoch
    shuffle = torch.randperm(trainData:size())
    -- do one epoch
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    for t = 1,trainData:size(),opt.batchSize do
        -- disp progress
        xlua.progress(t, trainData:size())
        -- create mini batch
        local inputs = {}
        local targets = {}
        for i = t,math.min(t+opt.batchSize-1,trainData:size()) do
            -- load new sample
            local input = trainData.data[shuffle[i]]
            local target = trainData.labels[shuffle[i]]
            input = input:cuda()
            table.insert(inputs, input)
            table.insert(targets, target)
        end
        -- create closure to evaluate f and df/dW
        local feval = 
            function(x)
                -- get new parameters
                if x ~= parameters then
                    parameters:copy(x)
                end
                -- reset gradients
                gradParameters:zero()
                -- f is the average of all criterions
                local f = 0
                -- evaluate function for complete mini batch
                for i = 1,#inputs do
                    -- estimate f
                    local output = theModel:forward(inputs[i])
                    local err = crit:forward(output, targets[i])
                    f = f + err
                    -- estimate df/dW
                    local df_do = crit:backward(output, targets[i])
                    --theModel:backward(inputs[i], df_do)
                    --weighted delta from next layer brought back times gradient of previous layer activation:
                    local GradWrtInput = theModel:updateGradInput(inputs[i],df_do)
                    --GradWrtInput at next layer times the previous layer activation:
                    theModel:accGradParameters(inputs[i],df_do)
                    -- update confusion
                    confusion:add(output, targets[i])
                end
                -- normalize gradients and f
                gradParameters:div(#inputs)
                f = f/#inputs
                -- return f and df/dW
                return f,gradParameters
            end
        -- optimize on current mini-batch; update parameters as per gradParameters as per
        -- parameters = parameters - learningRate * gradParameters
        optimMethod(feval, parameters, optimState)
    end
    -- time taken
    time = sys.clock() - time
    time = time / trainData:size()
    print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
    -- print confusion matrix
    print(confusion)
    -- update logger/plot
    --totalValid is the sum of the diagonal of the confusion matrix divided by the sum of the matrix. 
    --averageValid is the average of all diagonals divided by their respective rows.
    trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
    trainLogger:style{['% mean class accuracy (train set)'] = '-'}
    trainLogger:plot()
    -- save/log current net
    local filename = paths.concat(opt.save, 'svhnmodel.net')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    print('==> saving model to '..filename)
    torch.save(filename, model)
    -- next epoch
    confusion:zero()
    epoch = epoch + 1
end

manualTrainModel = function (theModel,crit)
    epoch = epoch or 1
    -- local vars
    local parameters,gradParameters = theModel:getParameters()
    local time = sys.clock()
    -- set model to training mode (for modules that differ in training and testing, like Dropout)
    theModel:training()
    -- shuffle at each epoch
    shuffle = torch.randperm(trainData:size())
    -- do one epoch
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    for t = 1,trainData:size(),opt.batchSize do
        -- disp progress
        xlua.progress(t, trainData:size())
        -- create mini batch
        local inputs = {}
        local targets = {}
        for i = t,math.min(t+opt.batchSize-1,trainData:size()) do
            -- load new sample
            local input = trainData.data[shuffle[i]]
            local target = trainData.labels[shuffle[i]]
            input = input:cuda()
            table.insert(inputs, input)
            table.insert(targets, target)
        end
        -- create closure to evaluate f and df/dW
        local feval = 
            function()
                -- reset gradients
                gradParameters:zero()
                -- f is the average of all criterions
                local f = 0
                -- evaluate function for complete mini batch
                for i = 1,#inputs do
                    -- estimate f
                    local output = theModel:forward(inputs[i])
                    local err = crit:forward(output, targets[i])
                    f = f + err
                    -- estimate df/dW
                    local df_do = crit:backward(output, targets[i])
                    --weighted delta from next layer brought back times gradient of previous layer activation:
                    local GradWrtInput = theModel:updateGradInput(inputs[i],df_do)
                    --GradWrtInput at next layer times the previous layer activation:
                    theModel:accGradParameters(inputs[i],df_do)
                    -- update confusion
                    confusion:add(output, targets[i])
                end
                -- normalize gradients and f
                gradParameters:div(#inputs)
                f = f/#inputs
            end
        -- update parameters using gradParameters as per
        -- parameters = parameters - learningRate * gradParameters
        feval()
        theModel:updateParameters(opt.learningRate)
    end
    -- time taken
    time = sys.clock() - time
    time = time / trainData:size()
    print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
    -- print confusion matrix
    print(confusion)
    -- update logger/plot
    --totalValid is the sum of the diagonal of the confusion matrix divided by the sum of the matrix. 
    --averageValid is the average of all diagonals divided by their respective rows.
    trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
    trainLogger:style{['% mean class accuracy (train set)'] = '-'}
    trainLogger:plot()
    -- save/log current net
    local filename = paths.concat(opt.save, 'model.net')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    print('==> saving model to '..filename)
    torch.save(filename, model)
    -- next epoch
    confusion:zero()
    epoch = epoch + 1
end

-- test function
testModel = function (theModel)
    -- local vars
    local time = sys.clock()
    -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
    theModel:evaluate()
    -- test over test data
    print('==> testing on test set:')
    for t = 1,testData:size() do
        -- disp progress
        xlua.progress(t, testData:size())
        -- get new sample
        local input = testData.data[t]
        input = input:cuda()
        local target = testData.labels[t]
        -- test sample
        local pred = theModel:forward(input)
        confusion:add(pred, target)
    end
    -- timing
    time = sys.clock() - time
    time = time / testData:size()
    print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')
    -- print confusion matrix
    print(confusion)
    -- update log/plot
    testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
    testLogger:style{['% mean class accuracy (test set)'] = '-'}
    testLogger:plot()
    -- next iteration:
    confusion:zero()
end

checkNumericalGrad = function(theModel,crit)
    -- local vars
    local computeCost = function(formodel,forcrit)
        local cost = 0
        for i = 1,trainData:size() do          
            local input = trainData.data[i]
            local target = trainData.labels[i]
            input = input:cuda()
            local output = formodel:forward(input)
            local err = forcrit:forward(output,target)
            cost = cost + err
        end
        return cost/trainData:size()
    end
    local parameters,gradParameters = theModel:getParameters()
    -- make a copy before modifying
    local copy_params = torch.zeros(parameters:size())
    copy_params:copy(parameters)
    local numgrad = torch.zeros(parameters:size());
    local perturb = torch.zeros(parameters:size());
    local e = 1e-4;
    for p = 1,parameters:numel() do
        xlua.progress(p, parameters:numel())
        -- Set perturbation vector
        perturb[p] = e;
        parameters = copy_params - perturb;
        local partialloss1_forp = computeCost(theModel,crit);
        parameters = copy_params + perturb;
        local partialloss2_forp = computeCost(theModel,crit);
        -- Compute Numerical Gradient
        numgrad[p] = (partialloss2_forp - partialloss1_forp) / (2*e);
        perturb[p] = 0;
    end
    -- copy back
    parameters:copy(copy_params)
    --
    local diff = gradParameters - numgrad
    print (diff:max(),diff:sum(),diff:mean())
end

loadSVHNData()
preprocessData()
verifyStats()
nnet,criterion = 
    buildConvNet(nfeaturemaps,nRowsOrCols,nfilterKernelsByLayer,nfiltsize,npoolsize,nMLPHiddenUnits,noutputs)
print(nnet,criterion)
while true do
    --trainModel(nnet,criterion)
    manualTrainModel(nnet,criterion)
    testModel(nnet)
end

