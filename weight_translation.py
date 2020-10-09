import torch
import os
import sys

weight_files = sys.argv[1:]

for weight_file in weight_files:
    try:
        assert os.path.isfile(weight_file)
        weights = torch.load(weight_file, map_location='cpu')
        temp = weights['model']
        # heads.bnneck.{weight, bias, running_mean, running_var, num_batches_tracked}
        # or heads.bnneck.0.{weight, bias, running_mean, running_var, num_batches_tracked}
        # to
        # heads.bottleneck.bnneck.{weight, running_mean, bias, running_var}
        try:
            temp['heads.bottleneck.bnneck.weight'] = temp['heads.bnneck.weight']
            temp['heads.bottleneck.bnneck.bias'] = temp['heads.bnneck.bias']
            temp['heads.bottleneck.bnneck.running_mean'] = temp['heads.bnneck.running_mean']
            temp['heads.bottleneck.bnneck.running_var'] = temp['heads.bnneck.running_var']
            temp['heads.bottleneck.bnneck.num_batches_tracked'] = temp['heads.bnneck.num_batches_tracked']
            print('convert heads.bnneck to heads.bottleneck.bnneck')
            try:
                temp.__delitem__('heads.bnneck.weight')
                temp.__delitem__('heads.bnneck.bias')
                temp.__delitem__('heads.bnneck.running_mean')
                temp.__delitem__('heads.bnneck.running_var')
                temp.__delitem__('heads.bnneck.num_batches_tracked')
            except:
                print('Fail in remove items')
        except:
            temp['heads.bottleneck.bnneck.weight'] = temp['heads.bnneck.0.weight']
            temp['heads.bottleneck.bnneck.bias'] = temp['heads.bnneck.0.bias']
            temp['heads.bottleneck.bnneck.running_mean'] = temp['heads.bnneck.0.running_mean']
            temp['heads.bottleneck.bnneck.running_var'] = temp['heads.bnneck.0.running_var']
            temp['heads.bottleneck.bnneck.num_batches_tracked'] = temp['heads.bnneck.0.num_batches_tracked']
            print('convert heads.bnneck.0 to heads.bottleneck.bnneck')
            try:
                temp.__delitem__('heads.bnneck.0.weight')
                temp.__delitem__('heads.bnneck.0.bias')
                temp.__delitem__('heads.bnneck.0.running_mean')
                temp.__delitem__('heads.bnneck.0.running_var')
                temp.__delitem__('heads.bnneck.0.num_batches_tracked')
            except:
                print('Fail in remove items')

        weights['model'] = temp
        with open(weight_file.strip('.pth') + '_translate.pth', 'wb') as f:
            torch.save(weights, f)
        print(f'translation {weight_file} completed, model weight is saved')
    except:
        print('Some faults in weight translation')