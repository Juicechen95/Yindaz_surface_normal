require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'optim'

require 'BatchIterator'
require 'utils'


-- config
local config = dofile('config.lua')
-- print(arg)
config = config.parse(arg)
-- print(config)
cutorch.setDevice(config.gpuid)
print("Start: " .. config.ps)

-- model
if config.model == 'fcn32s' then
    print('build fcn32s')
    local nn = require 'nn'
    model = nn.Sequential()
-- Keeping track of output dimensions
local W = 500
local H = 500
local inC = 3
local outC

-- Keeping track of convolution parameters
local kw --kernel width
local kh --kernel height
local dw --stride width
local dh --stride height
local pw --padding width
local ph --padding height
local group --grouping


model:add(nn.SpatialConvolution(3, 64, 3, 3, 1, 1, 100, 100))
model:add(nn.ReLU(true))

model:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))

model:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())

model:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))

model:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))

model:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())

model:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))

model:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))

model:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))

model:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())

model:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))

model:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))

model:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))

model:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())

model:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))

model:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))

model:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))

model:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())

model:add(nn.SpatialConvolution(512, 4096, 7, 7, 1, 1, 0, 0))
model:add(nn.ReLU(true))
model:add(nn.Dropout(0.5))

model:add(nn.SpatialConvolution(4096, 4096, 1, 1, 1, 1, 0, 0))
model:add(nn.ReLU(true))
model:add(nn.Dropout(0.5))

model:add(nn.SpatialConvolution(4096, 40, 1, 1, 1, 1, 0, 0))
--model:add(nn.SpatialFullConvolution(40,3,32,32,32,32):noBias())
model:add(nn.SpatialFullConvolution(40,3,64,64,32,32):noBias())--the output is 288*384
model:add(nn.Narrow(3,24,240))--the offset is different from FCN_seg(19)
model:add(nn.Narrow(4,32,320))
model:cuda()
parameters, gradParameters = model:getParameters()
else 
    print('iport model')
    model = dofile(config.model)(config) --nngraph
    parameters, gradParameters = model:getParameters()
    model:cuda()
    parameters, gradParameters = model:getParameters()
end
--print(model)
--graph.dot(model.fg, 'MLP')
--torch.save("./visulization/train_net.png", graph.dot(model.fg, 'MLP'))

-- resume training
if config.resume_training then
    print('loading saved model weight...')
    parameters:copy(torch.load(config.saved_model_weights))
    --config.optim_state = torch.load(config.saved_optim_state)
end

if config.finetune then
    print('finetune from saved model weight...')
    parameters:copy(torch.load(config.finetune_model))
    print('set up learning rate...')
    config.optim_state.learningRate = config.finetune_init_lr
end

-- criterion
local criterion_n = nn.CosineEmbeddingCriterion():cuda()

-- dataset
local train_data = loadData(config.train_file, config)
local test_data  = loadData(config.test_file, config)
local batch_iterator = BatchIterator(config, train_data, test_data)
batch_iterator:setBatchSize(config.batch_size)
-- logger
local logger = optim.Logger(config.ps .. '/log/train_loss', true)
local test_logger = optim.Logger(config.ps .. '/log/test_loss', true)
local losssum = 0.0
local N = 0
-- main training
for it_batch = 1, math.floor(config.nb_epoch * #batch_iterator.train.data / config.batch_size) do
    N = N +1
    --print(">>")
    local batch = batch_iterator:nextBatch('train', config)

    -- inputs and targets
    local inputs = batch.input
    inputs = inputs:contiguous():cuda()
    --print(inputs:size())
    
    local feval = function(x)
        -- prepare
        collectgarbage()
        if x ~= parameters then
            parameters:copy(x)
        end
        
        -- forward propagation
        --print('111')
        local est = model:forward(inputs)
        --print('est')
        --print(est:size())
        local valid = batch.valid
        valid = valid:cuda()
        local gnd = batch.output
        gnd = gnd:cuda()


        bz, ch, h, w = est:size(1), est:size(2), est:size(3), est:size(4)
        est = est:permute(1,3,4,2):contiguous():view(-1,ch)
        local normalize_layer = nn.Normalize(2):cuda()
        est_n = normalize_layer:forward(est)
        gnd = gnd:permute(1,3,4,2):contiguous():view(-1,ch)

        --print(est_n:size())
        --print(gnd:size())

        f = criterion_n:forward({est_n, gnd}, torch.Tensor(est_n:size(1)):cuda():fill(1))
        df = criterion_n:backward({est_n, gnd}, torch.Tensor(est_n:size(1)):cuda():fill(1))
        --print(df)
        df = df[1]
        --print(df:size())
        df = normalize_layer:backward(est, df)


        --print(valid:size())
        valid = valid:view(-1,1):expandAs(df)

        --print(valid:size())
        df[torch.eq(valid,0)] = 0

        df = df:view(-1, h, w, ch)
        df = df:permute(1, 4, 2, 3):contiguous()

        gradParameters:zero()
        model:backward(inputs, df)
        losssum = losssum + f
        -- print
        if it_batch % config.print_iters == 0 then
            print( it_batch, f)
        end

        -- log
        if it_batch % config.log_iters == 0 then
            logger:add{['normal_train_loss'] = (losssum/N)}
            N = 0
            losssum = 0.0
            logger:style{['% SN_train_loss'] = '-'}
            logger:plot()
            -- logger:add{['segmentation_loss'] = fs}
            -- logger:add{ f_normal, f_semantic, f_boundary, f_room}
            --logger:add{ f }
        end

        -- return
        -- return f_normal + f_semantic + f_boundary + f_room, gradParameters
        return f, gradParameters

    end

    -- optimizer
    optim.rmsprop(feval, parameters, config.optim_state)

    -- save
    if it_batch % config.snapshot_iters == 0 then
        print('saving model weight...')
        local filename
        filename = config.ps .. '/' .. 'iter_' .. it_batch .. '.t7'
        torch.save(filename, parameters)
        filename = config.ps .. '/' .. 'iter_' .. it_batch .. '_state.t7'
        torch.save(filename, config.optim_state)
    end

    -- lr
    if it_batch % config.lr_decay == 0 then
        config.optim_state.learningRate = config.optim_state.learningRate / config.lr_decay_t
        config.optim_state.learningRate = math.max(config.optim_state.learningRate, config.optim_state.learningRateMin)
        print('decresing lr... new lr:', config.optim_state.learningRate)
    end
    --print('one batch training finished!')

    --testing every batch
    local tag = it_batch
if tag % 795 == 0 then
    print(tag%50)
    tag = tag + 1
    print(tag%50)
    local test_count = 1
    local test_losssum = 0.0

    
    while test_count<= 654 do
        local test_batch = batch_iterator:nextBatch('test', config)
	    --local currName = batch_iterator:currentName('test')
        --print(currName)
	    --local k = split(currName, "/")
        --if config.matterport then
	        --saveName = k[#k-2] .. "_" .. k[#k]
        --else
            --saveName = k[#k-1] .. "_" .. k[#k]
        --end
	    --print(string.format("Testing %s", saveName))
	    local inputs = test_batch.input
        inputs = inputs:contiguous():cuda()
        local outputs = model:forward(inputs)
        local test_gnd = test_batch.output
        test_gnd = test_gnd:cuda()

        local ch, h, w = 0, 0, 0
        local normal_est, normal_mask, normal_gnd, f_normal, df_do_normal, normal_outputs = nil,nil,nil,nil,nil,nil

        normal_est = outputs
        ch, h, w = normal_est:size(2), normal_est:size(3), normal_est:size(4)
        normal_est = normal_est:permute(1, 3, 4, 2):contiguous()
        normal_est = normal_est:view(-1, ch)
        local normalize_layer = nn.Normalize(2):cuda()
        normal_outputs = normalize_layer:forward(normal_est)
        test_gnd = test_gnd:permute(1,3,4,2):contiguous():view(-1,ch)
        test_f = criterion_n:forward({normal_outputs, test_gnd}, torch.Tensor(normal_outputs:size(1)):cuda():fill(1))
	    --normal_outputs = normal_outputs:view(1, h, w, ch)
	    --normal_outputs = normal_outputs:permute(1, 4, 2, 3):contiguous()
	    --normal_outputs = normal_outputs:view( ch, h, w)
	    --normal_outputs = normal_outputs:float()
        test_losssum = test_losssum + test_f
        
        print(test_count)
        print(test_f)
        print(test_losssum/test_count)
	    --image.save(string.format("%s%s_normal_est.png", config.result_path, saveName), normal_outputs:add(1):mul(0.5))
        -- log
        if test_count % 654 == 0 then
            test_logger:add{['normal-test-loss'] = (test_losssum/test_count)}
            test_losssum = 0.0
            test_logger:style{['% SN-test-loss'] = '-'}
            test_logger:plot()
        end
        test_count = test_count + 1
        
    end
    
end

end

print('saving model weight...')
local filename
filename = config.ps .. 'final' .. '.t7'
torch.save(filename, parameters)
filename = config.ps .. 'final' .. '_state.t7'
torch.save(filename, config.optim_state)
