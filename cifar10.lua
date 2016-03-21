require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'xlua'
require 'image'
-- project files here:
require 'loadcifar10.lua'
require 'modelcifar10.lua'
--cifar10 package:
cifar10 = {}

cifar10.initialize = function ()
    --graphics servers
    cifar10.gfx = require 'gfx.js'
    cifar10.gfx.startserver(8501) --start server on port 8501
    cifar10.dygraph = require 'display' --start server in a separate session on fixed port 8000 using $th -ldisplay.start
    -- cmd and options
    local cmd = torch.CmdLine()
    cmd:option('-savePath', '/home/mit/projects/thtests/results', 'subdirectory to save/log experiments in')
    cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
    cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
    cmd:option('-trainingEpoch', 0, 'training iteration start value')
    cmd:option('-useOptimizer', true, 'whether to use optimizer(SGD) or to go manual')
    cifar10.options = cmd:parse(arg or {})
    print('processing options ==>',cifar10.options)
    -- input and model dims:
    -- input channels or feature maps
    cifar10.nfeaturemaps = 3
    cifar10.nRowsOrCols = 32 --image is 32 x 32
    -- filter sizes
    cifar10.nfilterKernelsByLayer = {64,64}
    cifar10.nfiltsize = 5
    cifar10.npoolsize = 2
    --mlp hidden units
    cifar10.nMLPHiddenUnits = {128,64}
    -- 10-class problem
    cifar10.noutputs = 10
    -- classes
    cifar10.classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
    -- optimizer objects: confusion matrix, loggers, optimizer method and state
    -- the confusion matrix records the current confusion across classes
    -- confusion:add(predicted,label) ==> confusion[label][predicted] = confusion[label][predicted] + 1
    -- target label is "printed" on rows, predicted counts are shown on columns
    -- ideally only diagonal elements should get updated i.e target label==predicted output
    cifar10.confusionMatrix = optim.ConfusionMatrix(cifar10.classes)
    -- log results to files
    cifar10.trainChartName = 'train'
    cifar10.testChartName = 'test'
    cifar10.trainLog = optim.Logger(paths.concat(cifar10.options.savePath,cifar10.trainChartName..'.log'))
    cifar10.testLog = optim.Logger(paths.concat(cifar10.options.savePath,cifar10.testChartName..'.log'))
    -- optimizer method
    cifar10.SGDOptimize = optim.sgd
    cifar10.optimizerState = {
        learningRate=cifar10.options.learningRate,
        weightDecay=0,
        momentum=0,
        learningRateDecay=1e-7}
    -- train and test dataset loading details
    cifar10.trainFile = 'cifar10-train.t7'
    cifar10.testFile = 'cifar10-test.t7'
    cifar10.downloadCommand = 'wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip'
    cifar10.unzipCommand = 'unzip cifar10torchsmall.zip'
    -- file to save/load model
    cifar10.modelfile = 'cifarmodel.net'
end

cifar10.createDatasets = function ()
    -- load datasets:
    cifar10.trainSet,cifar10.testSet = loadcifar10.loadData(
        cifar10.trainFile,
        cifar10.testFile,
        cifar10.downloadCommand,
        cifar10.unzipCommand)
    -- normalize datasets
    loadcifar10.normalizeData(cifar10.trainSet)
end

cifar10.createModel = function (usePersistedModel)
    local lastSavedModel = paths.concat(cifar10.options.savePath, cifar10.modelfile)
    if usePersistedModel and paths.filep(lastSavedModel) then
        cifar10.model = torch.load(lastSavedModel)
        cifar10.criterion = modelcifar10.buildCriterion()
        print 'loaded pre-trained model from disk and created new criterion'
    else
        --build model and criterion
        cifar10.model,cifar10.criterion = modelcifar10.buildConvNet(
            cifar10.nfeaturemaps,
            cifar10.nRowsOrCols,
            cifar10.nfilterKernelsByLayer,
            cifar10.nfiltsize,
            cifar10.npoolsize,
            cifar10.nMLPHiddenUnits,
            cifar10.noutputs)
        print 'created new model and criterion'
    end
end

cifar10.trainAndValidate = function (maxTrainingEpochs)
    for i =1,maxTrainingEpochs do
        --train the model
        modelcifar10.train(
            cifar10.trainSet,
            cifar10.model,
            cifar10.criterion,
            cifar10.options,
            cifar10.confusionMatrix,
            cifar10.trainChartName,
            cifar10.trainLog,
            cifar10.SGDOptimize,
            cifar10.optimizerState,
            cifar10.modelfile)
        --test the model
        modelcifar10.test(
            cifar10.testSet,
            cifar10.model,
            cifar10.confusionMatrix,
            cifar10.testChartName,
            cifar10.testLog)
    end
end

cifar10.setup = function ()
    cifar10.initialize()
    cifar10.createDatasets()
end

cifar10.run = function (usePersistedModel,maxTrainingEpochs)
    cifar10.createModel(usePersistedModel)
    cifar10.trainAndValidate(maxTrainingEpochs)
    cifar10.plotNVD3()
    cifar10.plotDyGraph()
end

cifar10.printmodel = function (model)
    for key,value in pairs(model) do
        print(key)
    end
    for key,value in pairs(model.modules) do
        print(key,value)
    end
    print(model.modules[1])
    cifar10.gfx.image(cifar10.model.modules[4].weight)
end

cifar10.plotNVD3 = function ()
    --plot mean class accuracy for the training and test sets
    local data = {
        {
            key    = 'Mean Training Accuracy',
            values = torch.Tensor(cifar10.trainLog.symbols[cifar10.trainChartName]),
            color  = '#ff7f0e',
        },
        {
            key = 'Mean Test Accuracy',
            values = torch.Tensor(cifar10.testLog.symbols[cifar10.testChartName]),
            color  = '#2ca02c',
        }
    }
    local config = {
        chart = 'line', 
        width = 640,
        height = 480,
        xLabel = "iterations(#)",
        yLabel = "accuracy(%)",
        useInteractiveGuideline = true
    }
    cifar10.gfx.chart(data,config)
end

cifar10.plotDyGraph = function ()
    local trainacc = {}
    for i,v in pairs(cifar10.trainLog.symbols[cifar10.trainChartName]) do
        table.insert(trainacc,{i,v})
    end
    local testacc = {}
    for i,v in pairs(cifar10.testLog.symbols[cifar10.testChartName]) do
        table.insert(testacc,{i,v})
    end
    --plot mean class accuracy for the training and test sets
    cifar10.dygraph.plot(trainacc,{labels={'epoch','accuracy'},title='training acc'})
    cifar10.dygraph.plot(testacc,{labels={'epoch','accuracy'},title='test acc'})
end

cifar10.visualizeImage = function (dataSet,sampleNum)
    cifar10.gfx.image(dataSet.data[sampleNum])
    cifar10.dygraph.image(dataSet.data[sampleNum])
end

cifar10.trySingle = function (dataSet,sampleNum)
    cifar10.visualizeImage(dataSet,sampleNum)
    modelcifar10.testSingleImage(dataSet,cifar10.model,cifar10.classes,sampleNum)
end

cifar10.main = function (usePersistedModel,maxTrainingEpochs)
    cifar10.setup()
    cifar10.run(usePersistedModel,maxTrainingEpochs)
end

return cifar10

