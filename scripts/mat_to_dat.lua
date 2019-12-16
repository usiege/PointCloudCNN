-- Usage: for converting .mat graphs in data/raw_data into .dat format read by torch
-- require 'matio' installed
-- *author: Muhan Zhang, Washington University in St. Louis

--require 'debug'
matio = require 'matio'
require 'paths'
require 'debug'
require 'torch'

Datasets = {'MUTAG', 'DD', 'NCI1', 'ptc', 'proteins', 'COLLAB', 'IMDBBINARY', 'IMDBMULTI'}

local dataset = torch.load('/media/charles/exFAT/working/PointCloud/scripts/dgcnn_lua/data'..'/'..'MUTAG'..'.dat')
print(dataset['instance'][188][2])

--[[
for _, dataname in pairs(Datasets) do
   print(dataname)
   local instance = {}
   local label = {}
   local datapath = '../data/raw_data/'
   local datapath = datapath..dataname..'.mat'

   local datapath = '/media/charles/exFAT/working/PointCloud/scripts/dgcnn_lua/data/raw_data/MUTAG.mat'
   --print(package.path)
   print(datapath)

--   local info = debug.getinfo(matio)
--   for k,v in pairs(info) do
--      print(k, ':', info[k])
--   end

   --print(debug.traceback())
   --local tmp = matio.load(datapath, {'lmutag'})
   --matio.use_lua_strings = true
   local tmp = matio.load(datapath, string.lower('l'..dataname))
   print(tmp)
   print(type(tmp))

   print('-----')
end
--]]
