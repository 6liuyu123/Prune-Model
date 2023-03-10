from models.models import ModelBuilder
from nni.algorithms.compression.v2.pytorch.pruning import L1NormPruner, LevelPruner
from nni.compression.pytorch import ModelSpeedup
from opts import get_parameters
from torch import nn
from torchsummary import summary

import os
import time
import torch
import torch.nn.functional as F
import torchvision

device = torch.device("cpu")

'''
    TODO
    1. image model prune
    2. bc prune, use
    3. load dataset(liuyu)
    4. test accuracy of each prune(liuyu)
'''

# TODO
# dataset = torchvision.datasets.Kinetics(
#     root = "/mnt/g/k/k400",
#     frames_per_clip = 16,    # TODO
#     step_between_clips = 16    # TODO
# )

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)


class VisualNet(nn.Module):
    def __init__(self, original_resnet):
        super(VisualNet, self).__init__()
        layers = list(original_resnet.children())[0:-2]
        self.feature_extraction = nn.Sequential(*layers)
    def forward(self, x):
        x = self.feature_extraction(x)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = x.view(x.size(0), -1)
        return x

class AudioNet(nn.Module):
    def __init__(self, original_resnet):
        super(AudioNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size = 7, stride = 1, padding = 3, bias = False)
        self.conv1.apply(weights_init)
        layers = [self.conv1]
        layers.extend(list(original_resnet.children())[1: -2])
        self.feature_extraction = nn.Sequential(*layers)
        self.freq_conv1x1 = nn.Conv2d(3, 1, kernel_size=(1, 1), padding=0, stride=1)
        self.freq_conv1x1.apply(weights_init)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.permute(0,3,1,2)
        x = self.freq_conv1x1(x)
        x = x.permute(0,2,1,3)
        # x = F.adaptive_max_pool2d(x, (1,1))    # TODO
        # x = x.view(x.size(0), -1)    # TODO
        return x

def build_audio():
    pretrained = True
    original_resnet = torchvision.models.resnet18(pretrained)
    net = AudioNet(original_resnet)
    return net

def build_image():
    pretrained = True
    original_resnet = torchvision.models.resnet18(pretrained)
    net = VisualNet(original_resnet)
    return net

if __name__ == '__main__':
    args = get_parameters("Prune-Model")
    model = build_audio()
    if args.modeltype == 'image':
        model = build_image()    # TODO
    summary(model, (1, 48, 48), device="cpu")
    config_list = [{
        'sparsity': args.sparsity,
        'op_types': ['Conv2d']
    }]
    start = time.time()
    pruner = L1NormPruner(model, config_list)
    _, masks = pruner.compress()
    pruner._unwrap_model()
    ModelSpeedup(model, dummy_input = torch.ones((1, 1, 48, 48)), masks_file = masks).speedup_model()
    torch.save(model.state_dict(), './pruned/%s_model_%s.pth' % (args.modeltype, args.sparsity))
    end = time.time()
    print('Running time: %s Seconds' % (end - start))