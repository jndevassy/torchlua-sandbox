require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'xlua'
require 'image'
require 'gfx.js'
-- project files here:
require 'loadcifar10.lua'
require 'modelcifar10.lua'
--cifar10 package:
cifar10 = {}

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
-- This matrix records the current confusion across classes
-- confusion:add(predicted,label) ==> confusion[label][predicted] = confusion[label][predicted] + 1
-- target label is "printed" on rows, predicted counts are shown on columns
-- ideally only diagonal elements should get updated i.e target label==predicted output
cifar10.confusionMatrix = optim.ConfusionMatrix(cifar10.classes)
-- log results to files
cifar10.trainLog = nil
cifar10.testLog = nil
-- options (savePath,batchSize,learningRate,trainingEpoch,maxEpochs,useOptimizer)
cifar10.options = nil
-- train and test dataSets
cifar10.trainSet,cifar10.testSet = nil,nil
cifar10.trainFile = 'cifar10-train.t7'
cifar10.testFile = 'cifar10-test.t7'
cifar10.downloadCommand = 'wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip'
cifar10.unzipCommand = 'unzip cifar10torchsmall.zip'
-- optimizer method
cifar10.SGDOptimize = optim.sgd

cifar10.setup = function ()
    gfx.startserver()
    local cmd = torch.CmdLine()
    cmd:option('-savePath', '/home/mit/projects/thtests/results', 'subdirectory to save/log experiments in')
    cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
    cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
    cmd:option('-trainingEpoch', 0, 'training iteration start value')
    cmd:option('-maxEpochs', 50, 'max training epochs')
    cmd:option('-useOptimizer', true, 'whether to use optimizer(SGD) or to go manual')
    cifar10.options = cmd:parse(arg or {})
    print('processing options ==>',cifar10.options)
    cifar10.trainLog = optim.Logger(paths.concat(cifar10.options.savePath, 'train.log'))
    cifar10.testLog = optim.Logger(paths.concat(cifar10.options.savePath, 'test.log'))
    cifar10.trainSet,cifar10.testSet = loadcifar10.loadData(
        cifar10.trainFile,
        cifar10.testFile,
        cifar10.downloadCommand,
        cifar10.unzipCommand)
    --loadcifar10.visualizeImage(cifar10.trainSet,33) --before
    loadcifar10.normalizeData(cifar10.trainSet)
    --loadcifar10.visualizeImage(cifar10.trainSet,33) --after
    --build model and criterion
    cifar10.model,cifar10.criterion = modelcifar10.buildConvNet(
        cifar10.nfeaturemaps,
        cifar10.nRowsOrCols,
        cifar10.nfilterKernelsByLayer,
        cifar10.nfiltsize,
        cifar10.npoolsize,
        cifar10.nMLPHiddenUnits,
        cifar10.noutputs)
end

cifar10.loop = function ()
    for i =1,cifar10.options.maxEpochs do
        --train the model
        modelcifar10.train(
            cifar10.trainSet,
            cifar10.model,
            cifar10.criterion,
            cifar10.options,
            cifar10.confusionMatrix,
            cifar10.trainLog,
            cifar10.SGDOptimize)
        --test the model
        modelcifar10.test(
            cifar10.testSet,
            cifar10.model,
            cifar10.confusionMatrix,
            cifar10.testLog)
    end
end

cifar10.main = function ()
    cifar10.setup()
    cifar10.loop()
end

cifar10.visualizeImage = loadcifar10.visualizeImage

cifar10.trySingle = function (dataSet,sampleNum)
    modelcifar10.testSingleImage(dataSet,cifar10.model,cifar10.classes,sampleNum)
end

return cifar10

