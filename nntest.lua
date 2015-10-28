require 'torch'
require 'nn'
require 'image'
require 'gfx.js'

gfx.startserver()

--[[command line arguments, example '$> th nntest.lua --batchSize 128 --momentum 0.5' ]]--
if not opt then
   cmd = torch.CmdLine()
   cmd:option('-batchSize', 128, 'batchSize')
   cmd:option('-momentum', 0.5, 'momentum')
   opt = cmd:parse(arg or {})
   print('processing options ==>',opt)
end

train_file = 'train_32x32.t7'
test_file = 'test_32x32.t7'
trainSize = 10000
testSize = 2000
channels = {'y','u','v'}

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
    mean = {}
    std = {}
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
    neighborhood = image.gaussian1D(13)
    -- Define our local normalization operator (It is an actual nn module, 
    -- which could be inserted into a trainable model):
    -- SpatialContrastiveNormalization(nInputPlane, kernel, threshold, thresval)
    normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()
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
        trainMean = trainData.data[{ {},i }]:mean()
        trainStd = trainData.data[{ {},i }]:std()
        testMean = testData.data[{ {},i }]:mean()
        testStd = testData.data[{ {},i }]:std()
        print('training data, '..channel..'-channel, mean: ' .. trainMean)
        print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)
        print('test data, '..channel..'-channel, mean: ' .. testMean)
        print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
    end
end

visualizeSamples = function (numSamples)
    samples_y = trainData.data[{ {1,numSamples},1 }]
    samples_u = trainData.data[{ {1,numSamples},2 }]
    samples_v = trainData.data[{ {1,numSamples},3 }]
    gfx.image(samples_y)
    gfx.image(samples_u)
    gfx.image(samples_v)
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
    return net
end

testnet = buildLeNet(1,32)
criterion = nn.CrossEntropyCriterion()


