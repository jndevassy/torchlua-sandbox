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

--cifar10 package:
cifar10 = {}

local main = function ()
    gfx.startserver()
    local tr,te = loadcifar10.loadData()
    cifar10["tr"] = tr
    cifar10["te"] = te    
    loadcifar10.normalizeData(tr)
end

cifar10["main"] = main
cifar10["loadcifar10"] = loadcifar10

return cifar10

