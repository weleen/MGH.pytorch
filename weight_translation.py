import torch

print('input weight file path')
weight_file = input()

try:
    weights = torch.load(weight_file, map_location='cpu')
    temp = weights['model']
    # heads.bnneck.{weight, bias, running_mean, running_var, num_batches_tracked}
    # to
    # heads.bnneck.0.{weight, running_mean, bias, running_var}
    temp['heads.bnneck.0.weight'] = temp['heads.bnneck.weight']
    temp['heads.bnneck.0.bias'] = temp['heads.bnneck.bias']
    temp['heads.bnneck.0.running_mean'] = temp['heads.bnneck.running_mean']
    temp['heads.bnneck.0.running_var'] = temp['heads.bnneck.running_var']
    temp['heads.bnneck.0.num_batches_tracked'] = temp['heads.bnneck.num_batches_tracked']
    weights['model'] = temp
    print('translation completed')
    with open(weight_file[:-10] + '_translate.pth', 'wb') as f:
        torch.save(weights, f)
    print('model weight is saved')
except:
    print('Some faults in weight translation')