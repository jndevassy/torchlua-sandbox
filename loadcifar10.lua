
--loadcifar10 package:
loadcifar10 = {}

local loadData = function ()
    local trainFile = 'cifar10-train.t7'
    local testFile = 'cifar10-test.t7'
    if not paths.filep(trainFile) or not paths.filep(testFile) then
        os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
        os.execute('unzip cifar10torchsmall.zip')
    end
    local trainData = torch.load(trainFile)
    trainData.data = trainData.data:float()
    trainData["size"] = function(self) return self.data:size(1) end
    local testData = torch.load(testFile)
    testData.data = testData.data:float()
    testData["size"] = function(self) return self.data:size(1) end
    return trainData,testData
end

local normalizeData = function (dataSet)
    --dataSet ==> Samples x Channels x Height(rows) x Width(cols)
    local mean = {}
    local std = {}
    for c = 1,dataSet.data:size(2) do
        mean[c] = dataSet.data[{ {},{c},{},{} }]:mean()
        std[c] = dataSet.data[{ {},{c},{},{} }]:std()
        dataSet.data[{ {},{c},{},{} }]:add(-mean[c])
        dataSet.data[{ {},{c},{},{} }]:div(std[c])
        print('norm mean for channel '..c..' is '..dataSet.data[{ {},{c},{},{} }]:mean())
        print('norm std for channel '..c..' is '..dataSet.data[{ {},{c},{},{} }]:std())
    end
end

loadcifar10["loadData"] = loadData
loadcifar10["normalizeData"] = normalizeData

return loadcifar10

