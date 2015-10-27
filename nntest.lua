require 'nn'
require 'dp'

--[[command line arguments, example '$> th nntest.lua --batchSize 128 --momentum 0.5' ]]--
cmd = torch.CmdLine()
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--lrDecay', 'linear', 'type of learning rate decay : adaptive | linear | schedule | none')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--saturateEpoch', 300, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--schedule', '{}', 'learning rate schedule')
cmd:option('--maxWait', 4, 'maximum number of epochs to wait for a new minima to be found. After that, the learning rate is decayed by decayFactor.')
cmd:option('--decayFactor', 0.001, 'factor by which learning rate is decayed for adaptive decay.')
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--momentum', 0, 'momentum')
cmd:option('--hiddenSize', '{200,200}', 'number of hidden units per layer')
cmd:option('--batchSize', 32, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--dropout', false, 'apply dropout on hidden neurons')
cmd:option('--batchNorm', false, 'use batch normalization. dropout is mostly redundant with this')
cmd:option('--dataset', 'Mnist', 'which dataset to use : Mnist | NotMnist | Cifar10 | Cifar100')
cmd:option('--standardize', false, 'apply Standardize preprocessing')
cmd:option('--zca', false, 'apply Zero-Component Analysis whitening')
cmd:option('--progress', false, 'display progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')
opt = cmd:parse(arg or {})
opt.schedule = dp.returnString(opt.schedule)
opt.hiddenSize = dp.returnString(opt.hiddenSize)
if not opt.silent then
   table.print(opt)
end
--

BuildLeNet = function (inputChannels,inputSize)
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
    return net
end

testnet = BuildLeNet(1,32)
criterion = nn.CrossEntropyCriterion()

input_preprocess = {}
table.insert(input_preprocess, dp.Standardize())
ds = dp.Cifar10{input_preprocess = input_preprocess}






--[[
input_ = torch.rand(1,32,32) --an image of the digit 3
target = 3 --class for the digit 3
zOutput_ = testnet:forward(input_) --net prediction for image of digit 3, after softmax activation

Err = criterion:forward(zOutput_,target) --total network scalar error at the output => E = -sum[ d~j * log(y~j) ] or -torch.log(torch.exp(zOutput_[target])/torch.sum(torch.exp(zOutput_)))
dErr_dz_ = criterion:backward(zOutput_,target) --network error grad vector at the output wrt softmax activation input => dE/dz = -(d-y) or -d + torch.exp(zOutput_)/torch.sum(torch.exp(zOutput_))

testnet:zeroGradParameters()
gradInput_ = testnet:backward(input_,dErr_dz_) --backprop using chain rule
testnet:updateParameters(learningRate)
]]--




