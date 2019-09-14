from torchvision.models.densenet import *


# Thanks for the model: https://github.com/facebookresearch/WSL-Images/blob/master/hubconf.py
def _resnext(path, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    checkpoint = torch.load(path)
#     epoch = checkpoint['epoch'];    best_prec = checkpoint['best_prec'];    loss_train = checkpoint['loss_train']
    model.load_state_dict(checkpoint)
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     print("=> loaded checkpoint '{}' (epoch {})" .format(epoch, checkpoint['epoch']))
#     return epoch, best_prec, loss_train
#     state_dict = load_state_dict(model_urls[arch], progress=progress)
#    TODO try with different target
#     model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(2048, 1024), nn.LeakyReLU(inplace=True), nn.Linear(1024,67), nn.Softmax(dim=1))
    return model


def resnext101_32x16d_wsl(path, progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x16 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 16
    return _resnext(path, Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)

# shutil.copyfile('../input/densenet201/densenet201.pth', '/tmp/.cache/torch/checkpoints/densenet201-c1103571.pth')

def densenet201(path=DENSENET101_PETRAINED_PATH, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), **kwargs)
    model.load_state_dict(torch.load(path))
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['densenet201']))
    return model
