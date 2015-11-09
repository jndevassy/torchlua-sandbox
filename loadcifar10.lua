--loadcifar10 package:
loadcifar10 = {}

loadcifar10.loadData = function (trainFile,testFile,downloadCmd,unzipCmd)
    if not paths.filep(trainFile) or not paths.filep(testFile) then
        if downloadCmd then os.execute(downloadCmd) end
        if unzipCmd then os.execute(unzipCmd) end
    end
    local trainData = torch.load(trainFile)
    trainData.data = trainData.data:cuda()
    trainData["size"] = function(self) return self.data:size(1) end
    local testData = torch.load(testFile)
    testData.data = testData.data:cuda()
    testData["size"] = function(self) return self.data:size(1) end
    return trainData,testData
end

loadcifar10.normalizeData = function (dataSet)
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

return loadcifar10

