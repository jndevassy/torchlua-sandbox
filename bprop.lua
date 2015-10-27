require 'nn'

IU = 1; OU = 1; HU = 2; learningRate = 0.01;
fx = function(X)
    local y = X*2.0 + 0.5
    return y
end

mlp = nn.Sequential()
mlp:add(nn.Linear(IU,HU))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HU,OU))
params,gradParams = mlp:getParameters()
print(params,gradParams)
crit = nn.MSECriterion()

trainExample = function()
    local X = torch.randn(1)
    local Y = fx(X)
    local Hyp = mlp:forward(X)
    local Loss = crit:forward(Hyp,Y)
    --delta at output:
    local GradWrtOutput = crit:backward(Hyp,Y) 
    --weighted delta from next layer brought back times gradient of previous layer activation:
    local GradWrtInput = mlp:updateGradInput(X,GradWrtOutput) 
    mlp:zeroGradParameters()
    --GradWrtInput at next layer times the previous layer activation:
    mlp:accGradParameters(X,GradWrtOutput) 
    mlp:updateParameters(learningRate)
    return Loss,GradWrtOutput,GradWrtInput
end

for iter = 1,10000 do
    trainExample()
end


