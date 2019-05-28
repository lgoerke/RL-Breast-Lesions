import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class RCombiNet(nn.Module):
    def __init__(self, pretrained_resnet, input_size, node_size_rec, node_size, no_actions):
        super(RCombiNet, self).__init__()
        self.feature_model = pretrained_resnet
        self.pretrained_resnet = pretrained_resnet
        self.input_size = input_size
        self.node_size_rec = node_size_rec
        self.node_size = node_size
        self.no_actions = no_actions

        self.conv1 = pretrained_resnet.conv1
        self.bn1 = pretrained_resnet.bn1
        self.relu = pretrained_resnet.relu
        self.maxpool = pretrained_resnet.maxpool
        self.layer1 = pretrained_resnet.layer1
        self.layer2 = pretrained_resnet.layer2
        self.layer3 = pretrained_resnet.layer3
        self.layer4 = pretrained_resnet.layer4
        self.ll1 = nn.Linear(input_size, node_size_rec)
        self.relu2 = nn.ReLU()
        self.lstm = nn.LSTM(node_size_rec, node_size)
        if torch.cuda.is_available():
            self.hidden_running = (torch.randn(1, 1, node_size).cuda(),
                                   torch.randn(1, 1, node_size).cuda())
            self.hidden_update = (torch.randn(1, 1, node_size).cuda(),
                                  torch.randn(1, 1, node_size).cuda())
        else:
            self.hidden_running = (torch.randn(1, 1, node_size),
                                   torch.randn(1, 1, node_size))
            self.hidden_update = (torch.randn(1, 1, node_size),
                                  torch.randn(1, 1, node_size))

        self.ll2 = nn.Linear(node_size, node_size)
        self.ll3 = nn.Linear(node_size, no_actions)


    def reset_hidden_state_running(self):
        del self.hidden_running
        if torch.cuda.is_available():
            self.hidden_running = (torch.randn(1, 1, self.node_size).cuda(),
                                   torch.randn(1, 1, self.node_size).cuda())
        else:
            self.hidden_running = (torch.randn(1, 1, self.node_size),
                                   torch.randn(1, 1, self.node_size))

    def reset_hidden_state_update(self):
        del self.hidden_update
        if torch.cuda.is_available():
            self.hidden_update = (torch.randn(1, 1, self.node_size).cuda(),
                                  torch.randn(1, 1, self.node_size).cuda())
        else:
            self.hidden_update = (torch.randn(1, 1, self.node_size),
                                  torch.randn(1, 1, self.node_size))

    def forward(self, x):
        x = self.ll1(x)
        x = self.relu2(x)
        inp = x.view(1, 1, -1)
        out, self.hidden_running = self.lstm(inp, self.hidden_running)
        x = self.ll2(out)
        x = self.relu2(x)
        x = self.ll3(x)
        x = x.view(x.size(0), -1)
        return x

    def get_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)
        return x

    def forward_all(self, x):
        self.reset_hidden_state_update()
        for sequence in x:
            sequence = self.get_features(sequence)
            sequence = self.ll1(sequence)
            sequence = self.relu2(sequence)
            inp = sequence.view(1, 1, -1)
            out, self.hidden_update = self.lstm(inp, self.hidden_update)
        x = self.ll2(out)
        x = self.relu2(x)
        x = self.ll3(x)
        x = x.view(x.size(0), -1)
        return x

    def forward_sequence(self, x):
        self.reset_hidden_state_update()
        for sequence in x:
            sequence = self.ll1(sequence)
            sequence = self.relu2(sequence)
            inp = sequence.view(1, 1, -1)
            out, self.hidden_update = self.lstm(inp, self.hidden_update)
        x = self.ll2(out)
        x = self.relu2(x)
        x = self.ll3(x)
        x = x.view(x.size(0), -1)
        return x



class CombiNet(nn.Module):
    def __init__(self, pretrained_resnet, input_size, node_size, no_actions, with_pool=False):
        super(CombiNet, self).__init__()

        self.feature_model = pretrained_resnet
        self.input_size = input_size
        self.node_size = node_size
        self.no_actions = no_actions
        self.with_pool = with_pool
        
        self.conv1 = pretrained_resnet.conv1
        self.bn1 = pretrained_resnet.bn1
        self.relu = pretrained_resnet.relu
        self.maxpool = pretrained_resnet.maxpool
        self.layer1 = pretrained_resnet.layer1
        self.layer2 = pretrained_resnet.layer2
        self.layer3 = pretrained_resnet.layer3
        self.layer4 = pretrained_resnet.layer4
        
        if self.with_pool:
            self.avgpool = pretrained_resnet.avgpool
        self.qnet = nn.Sequential(
            nn.Linear(input_size, node_size),
            nn.ReLU(),
            nn.Linear(node_size, node_size),
            nn.ReLU(),
            nn.Linear(node_size, no_actions)
        )

    def forward(self, x):
#         print('Forward {}'.format(x.shape))
#         print('Forward 01',torch.mean(x,0),flush=True)
#         print('Forward 01',torch.mean(x,1),flush=True)
        x = self.qnet(x)
#         print('Forward 02',torch.mean(x,0),flush=True)
#         print('Forward 02',torch.mean(x,1),flush=True)
        return x

    def get_features(self, x):
#         print('Get features {}'.format(x.shape))
#         print('Get features 01',torch.mean(x,0),flush=True)
#         print('Get features 01',torch.mean(x,1),flush=True)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
#         print('Get features 02',torch.mean(x,0),flush=True)
#         print('Get features 02',torch.mean(x,1),flush=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.with_pool:
            x = self.avgpool(x)
#         print('Get features 03',torch.mean(x,0),flush=True)
#         print('Get features 03',torch.mean(x,1),flush=True)
        x = x.view(x.size(0), -1)
        return x

    def forward_all(self, x):
#         print('Forward all {}'.format(x.shape))
#         print('Forward all 01',torch.mean(x.view(-1),0),flush=True)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
#         print('Forward all 02',torch.mean(x.view(-1),0),flush=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.with_pool:
            x = self.avgpool(x)
#         print('Forward all 03',torch.mean(x.view(-1),0),flush=True)
        x = x.view(x.size(0), -1)
        x = self.qnet(x)
#         print('Forward all 04',torch.mean(x.view(-1),0),flush=True)
        return x


class ResNetFeatures(nn.Module):
    def __init__(self, pretrained_resnet):
        super(ResNetFeatures, self).__init__()
        self.conv1 = pretrained_resnet.conv1
        self.bn1 = pretrained_resnet.bn1
        self.relu = pretrained_resnet.relu
        self.maxpool = pretrained_resnet.maxpool
        self.layer1 = pretrained_resnet.layer1
        self.layer2 = pretrained_resnet.layer2
        self.layer3 = pretrained_resnet.layer3
        self.layer4 = pretrained_resnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)

        return x
    
class ResNetFeaturesPool(nn.Module):
    def __init__(self, pretrained_resnet):
        super(ResNetFeaturesPool, self).__init__()
        self.conv1 = pretrained_resnet.conv1
        self.bn1 = pretrained_resnet.bn1
        self.relu = pretrained_resnet.relu
        self.maxpool = pretrained_resnet.maxpool
        self.layer1 = pretrained_resnet.layer1
        self.layer2 = pretrained_resnet.layer2
        self.layer3 = pretrained_resnet.layer3
        self.layer4 = pretrained_resnet.layer4
        self.avgpool = pretrained_resnet.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        return x


class ResNetAuto(nn.Module):
    def __init__(self, pretrained_resnet):
        super(ResNetAuto, self).__init__()
        self.conv1 = pretrained_resnet.conv1
        self.bn1 = pretrained_resnet.bn1
        self.relu = pretrained_resnet.relu
        self.maxpool = pretrained_resnet.maxpool
        self.layer1 = pretrained_resnet.layer1
        self.layer2 = pretrained_resnet.layer2
        self.layer3 = pretrained_resnet.layer3
        self.layer4 = pretrained_resnet.layer4
        self.deconv1 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1, groups=1, bias=True,
                                          dilation=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1, groups=1, bias=True,
                                          dilation=1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1, groups=1, bias=True,
                                          dilation=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1, groups=1, bias=True,
                                          dilation=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1, groups=1, bias=True,
                                          dilation=1)
        self.bn5 = nn.BatchNorm2d(3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.deconv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.deconv5(x)
        x = self.bn5(x)
        x = self.sigmoid(x)

        return x


def get_resnet_classification(pretrained=False, num_classes=1):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1000)
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        model.fc = nn.Linear(512, num_classes)
    else:
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return model

def get_resnet_features(given_model=None):
    if given_model:
        model = ResNetFeatures(given_model)
    else:
        pre_trained_model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1000)
        pre_trained_model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        model = ResNetFeatures(pre_trained_model)
    return model


def get_resnet_auto(given_model=None):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if given_model:
        model = ResNetAuto(given_model)
    else:
        pre_trained_model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1000)
        pre_trained_model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        model = ResNetAuto(pre_trained_model)
    return model
