import torch
import torch.nn as nn
import torchvision.models as m


class SimpleNet(nn.Module):
    def __init__(self, learn_pos=False,cat=False):
        super(SimpleNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )
        self.l1 = nn.Linear(128 * 13 * 13, 4096)
        self.relu = nn.ReLU()
        if learn_pos:
            self.decision = nn.Linear(4096, 3)
        elif cat:
            self.decision = nn.Linear(4096, 5)
        else:
            self.decision = nn.Linear(4096, 1)

    def forward(self, x):
        # print('00', x.shape, flush=True)
        x = self.features(x)
        # for module_pos, module in self.features._modules.items():
        #     # 7
        #     print('-',x.size(),flush=True)
        #     x = module(x)
        #     print('{}, {}:\n{}\n'.format(module_pos, self.features._modules[module_pos], x.shape), flush=True)
        # print('001 vor resize', x.shape, flush=True)
        x = x.view(x.size(0), -1)
        # print('002 vor linear', x.shape, flush=True)
        x = self.l1(x)
        x = self.relu(x)
        # print('003 vor decision', x.shape, flush=True)
        x = self.decision(x)
        # print('004 vor resize', x.shape, flush=True)
        x = x.view(x.size(0), -1)
        # print('005 output', x.shape, flush=True)
        return x


class RCombiNet(nn.Module):
    def __init__(self, featnet_no_decision, input_size, node_size_rec, node_size, no_actions):
        super(RCombiNet, self).__init__()
        self.features = featnet_no_decision.features
        
        self.feature_model = featnet_no_decision
        self.input_size = input_size
        self.node_size_rec = node_size_rec
        self.node_size = node_size
        self.no_actions = no_actions
        
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

    def forward_all(self, x):
        self.reset_hidden_state_update()
        for sequence in x:
            # print('-',sequence.shape, flush=True)
            inp = self.features(sequence)
            inp = inp.view(sequence.shape[0], -1)
            # print('--',inp.shape, flush=True)
            inp = self.ll1(inp)
            inp = self.relu2(inp)
            # print('---',inp.shape, flush=True)
            inp = inp.view(1, 1, -1)
            # print('---',inp.shape, flush=True)
            # print('Forward_all',inp.shape)
            out, self.hidden_update = self.lstm(inp, self.hidden_update)
        x = self.ll2(out)
        x = self.ll3(x)
        return x

    def forward_sequence(self, x):
        # print('X',x.shape)
        self.reset_hidden_state_update()
        for sequence in x:
            # print('seq',sequence.shape)
            sequence = self.ll1(sequence)
            sequence = self.relu2(sequence)
            inp = sequence.view(1, 1, -1)
            # print('Forward seq',inp.shape)
            out, self.hidden_update = self.lstm(inp, self.hidden_update)
            # print('Out {}, Hidden {}'.format(out, self.hidden_update))
        x = self.ll2(out)
        x = self.ll3(x)
        return x

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
        # x = self.features(x)
        x = self.ll1(x)
        x = self.relu2(x)
        inp = x.view(1, 1, -1)
        # print('Normal forward',inp.shape)
        out, self.hidden_running = self.lstm(inp, self.hidden_running)
        x = self.ll2(out)
        x = self.ll3(x)
        return x

    def get_features(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class CombiNet(nn.Module):
    def __init__(self, featnet_no_decision, input_size, node_size, no_actions):
        super(CombiNet, self).__init__()
        self.feature_model = featnet_no_decision
        self.input_size = input_size
        self.node_size = node_size
        self.no_actions = no_actions

        self.features = featnet_no_decision.features
        self.qnet = nn.Sequential(
            nn.Linear(input_size, node_size),
            nn.ReLU(),
            nn.Linear(node_size, node_size),
            nn.ReLU(),
            nn.Linear(node_size, no_actions)
        )

    def forward(self, x):
        x = self.qnet(x)
        return x

    def get_features(self, x):
        x = self.features(x)
        # for module_pos, module in self.features._modules.items():
        #     # 7
        #     # print('-',x.size(),flush=True)
        #     if isinstance(module, m.resnet.BasicBlock):
        #         for _module_pos, _module in module._modules.items():
        #             # print('Before {}, {}:\n{}\n'.format(_module_pos, module._modules[_module_pos], x.shape),
        #             #       flush=True)
        #             x = _module(x)
        #             # print('After {}, {}:\n{}\n'.format(_module_pos, module._modules[_module_pos], x.shape), flush=True)
        #             # print('+++++++\n')
        #     else:
        #         # print('Before {}, {}:\n{}\n'.format(module_pos, self.features._modules[module_pos], x.shape), flush=True)
        #         x = module(x)
        #         # print('After {}, {}:\n{}\n'.format(module_pos,self.features._modules[module_pos],x.shape),flush=True)
        #         # print('+++++++\n')
        x = x.view(x.size(0), -1)
        return x

    def forward_all(self, x):
        x = self.features(x)
        # for module_pos, module in self.features._modules.items():
        #     # 7
        #     # print('-',x.size(),flush=True)
        #     if isinstance(module, m.resnet.BasicBlock):
        #         for _module_pos, _module in module._modules.items():
        #             # print('Before {}, {}:\n{}\n'.format(_module_pos, module._modules[_module_pos], x.shape),
        #             #       flush=True)
        #             x = _module(x)
        #             # print('After {}, {}:\n{}\n'.format(_module_pos, module._modules[_module_pos], x.shape), flush=True)
        #             # print('+++++++\n')
        #     else:
        #         # print('Before {}, {}:\n{}\n'.format(module_pos, self.features._modules[module_pos], x.shape), flush=True)
        #         x = module(x)
        #         # print('After {}, {}:\n{}\n'.format(module_pos,self.features._modules[module_pos],x.shape),flush=True)
        #         # print('+++++++\n')
        x = x.view(x.size(0), -1)
        # x = Variable(torch.cat((x.data, history.data)))

        x = self.qnet(x)

        return x


class RQNet(nn.Module):
    def __init__(self, input_size, node_size, no_actions):
        super(RQNet, self).__init__()
        self.input_size = input_size
        self.node_size = node_size
        self.lstm = nn.LSTM(input_size, node_size)
        self.hidden_running = (torch.randn(1, 1, node_size),
                               torch.randn(1, 1, node_size))
        self.hidden_update = (torch.randn(1, 1, node_size),
                              torch.randn(1, 1, node_size))
        self.ll2 = nn.Linear(node_size, node_size)
        self.ll3 = nn.Linear(node_size, no_actions)

    def forward_seq(self, x):
        for sequence in x:
            inp = sequence.view(1, 1, -1)
            # print(inp.shape)
            out, self.hidden_update = self.lstm(inp, self.hidden_update)
        x = self.ll2(out)
        x = self.ll3(x)
        return x

    def reset_hidden_state_r(self):
        self.hidden_running = (torch.randn(1, 1, self.node_size),
                               torch.randn(1, 1, self.node_size))

    def reset_hidden_state_u(self):
        self.hidden_update = (torch.randn(1, 1, self.node_size),
                              torch.randn(1, 1, self.node_size))

    def forward(self, x):
        inp = x.view(1, 1, -1)
        # print(inp.shape)
        out, self.hidden_running = self.lstm(inp, self.hidden_running)
        x = self.ll2(out)
        x = self.ll3(x)
        return x


class QNet(nn.Module):
    def __init__(self, input_size, node_size, no_actions):
        super(QNet, self).__init__()

        self.ll1 = nn.Linear(input_size, node_size)
        self.ln1 = nn.LayerNorm(node_size)

        self.ll2 = nn.Linear(node_size, node_size)
        self.ln2 = nn.LayerNorm(node_size)

        self.ll3 = nn.Linear(node_size, no_actions)
        self.soft = nn.Softmax()

    def forward(self, x, add_noise=False, scale=None):
        if add_noise and scale:
            for param in self.parameters():
                param.data.copy_(
                    param.data + torch.normal(mean=torch.zeros(param.size), std=torch.ones(param.size) * scale))

        x = self.ll1(x)
        if add_noise and scale:
            x = self.ln1(x)

        x = self.ll2(x)
        if add_noise and scale:
            x = self.ln2(x)

        x = self.ll3(x)
        if add_noise and scale:
            x = self.soft(x)
        return x


class FCResNet(nn.Module):
    def __init__(self, pretrained_model, learn_pos=False):
        super(FCResNet, self).__init__()
        self.features = pretrained_model.features
        # Size after features part:
        # Num imgs x 512 x 7 x 7

        # Reduce to 4x4
        self.fc1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # Reduce to 2x2
        self.fc2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # Reduce to 1x1
        self.fc3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.decision = nn.Conv2d(512, 1, kernel_size=(1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.decision(x)
        x = x.view(x.size(0), -1)
        return x


class NetNoDecisionLayer(nn.Module):
    def __init__(self, pretrained_model):
        super(NetNoDecisionLayer, self).__init__()
        self.features = pretrained_model.features

    def forward(self, x):
        for module_pos, module in self.features._modules.items():
            # 7
            # print('-',x.size(),flush=True)
            if isinstance(module, m.resnet.BasicBlock):
                for _module_pos, _module in module._modules.items():
                    # print('Before {}, {}:\n{}\n'.format(_module_pos, module._modules[_module_pos], x.shape),
                    #       flush=True)
                    x = _module(x)
                    # print('After {}, {}:\n{}\n'.format(_module_pos, module._modules[_module_pos], x.shape), flush=True)
                    # print('+++++++\n')
            else:
                # print('Before {}, {}:\n{}\n'.format(module_pos, self.features._modules[module_pos], x.shape), flush=True)
                x = module(x)
                # print('After {}, {}:\n{}\n'.format(module_pos,self.features._modules[module_pos],x.shape),flush=True)
                # print('+++++++\n')
        x = x.view(x.size(0), -1)
        return x


class ModiefiedResNetLessFilter(nn.Module):
    def __init__(self, learn_pos=False):
        super(ModiefiedResNetLessFilter, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            # nn.Sequential(
            m.resnet.BasicBlock(16, 16, 1),
            m.resnet.BasicBlock(16, 16, 1),
            # ),
            # nn.Sequential(
            m.resnet.BasicBlock(16, 32, 2),
            m.resnet.BasicBlock(32, 32, 1),
            # ),
            # nn.Sequential(
            m.resnet.BasicBlock(32, 64, 2),
            m.resnet.BasicBlock(64, 64, 1),
            # ),
            # nn.Sequential(
            m.resnet.BasicBlock(64, 128, 2),
            m.resnet.BasicBlock(128, 128, 1)
            # )
        )
        self.avgpool = nn.AvgPool2d(7, stride=1)
        if learn_pos:
            self.decision = nn.Linear(128, 3)
        else:
            self.decision = nn.Linear(128, 1)

    def forward(self, x):
        # print('00',x.shape,flush=True)
        for module_pos, module in self.features._modules.items():
            # 7
            # print('-',x.size(),flush=True)
            if isinstance(module, m.resnet.BasicBlock):
                for _module_pos, _module in module._modules.items():
                    # print('Before {}, {}:\n{}\n'.format(_module_pos, module._modules[_module_pos], x.shape),
                    #       flush=True)
                    x = _module(x)
                    # print('After {}, {}:\n{}\n'.format(_module_pos, module._modules[_module_pos], x.shape), flush=True)
                    # print('+++++++\n')
            else:
                # print('Before {}, {}:\n{}\n'.format(module_pos, self.features._modules[module_pos], x.shape), flush=True)
                x = module(x)
                # print('After {}, {}:\n{}\n'.format(module_pos,self.features._modules[module_pos],x.shape),flush=True)
                # print('+++++++\n')
        x = self.avgpool(x)
        # print('000',x.shape,flush=True)
        x = x.view(x.size(0), -1)
        # print('001',x.shape,flush=True)
        x = self.decision(x)
        # print('002',x.shape,flush=True)
        x = x.view(x.size(0), -1)
        # print('003',x.shape,flush=True)
        return x


class ModifiedResNetAuto(nn.Module):
    def __init__(self, pretrained_model, learn_pos=False):
        super(ModifiedResNetAuto, self).__init__()
        self.features = nn.Sequential(
            pretrained_model.conv1,
            pretrained_model.bn1,
            pretrained_model.relu,
            pretrained_model.maxpool,
            pretrained_model.layer1,
            pretrained_model.layer2,
            pretrained_model.layer3,
            pretrained_model.layer4
        )

        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()
        self.deconv1 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1, groups=1, bias=True,
                                          dilation=1)
        self.deconv2 = nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1, groups=1, bias=True,
                                          dilation=1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1, groups=1, bias=True,
                                          dilation=1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1, groups=1, bias=True,
                                          dilation=1)
        self.deconv5 = nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1, groups=1, bias=True,
                                          dilation=1)
        # self.decision = nn.Linear(512, 1)

    def forward(self, x):
        print(torch.min(x))
        print(torch.max(x))
        print('00', x.shape, flush=True)
        for module_pos, module in self.features._modules.items():
            # 7
            # print('-',x.size(),flush=True)
            x = module(x)
            # print('{}, {}:\n{}\n'.format(module_pos, self.features._modules[module_pos], x.shape), flush=True)
        x = self.deconv1(x)
        print('001', x.shape, flush=True)
        x = self.relu(x)
        x = self.deconv2(x)
        print('002', x.shape, flush=True)
        x = self.relu(x)
        x = self.deconv3(x)
        print('003', x.shape, flush=True)
        x = self.relu(x)
        x = self.deconv4(x)
        print('004', x.shape, flush=True)
        x = self.relu(x)
        x = self.deconv5(x)
        print('005', x.shape, flush=True)
        print(torch.min(x))
        print(torch.max(x))
        x = self.sigm(x)*255
        print(torch.min(x))
        print(torch.max(x))
        return x


class ModifiedResNet(nn.Module):
    def __init__(self, pretrained_model, learn_pos=False):
        super(ModifiedResNet, self).__init__()
        self.features = nn.Sequential(
            pretrained_model.conv1,
            pretrained_model.bn1,
            pretrained_model.relu,
            pretrained_model.maxpool,
            pretrained_model.layer1,
            pretrained_model.layer2,
            pretrained_model.layer3,
            pretrained_model.layer4
        )

        self.avgpool = nn.AvgPool2d(7, stride=1)
        if learn_pos:
            self.decision = nn.Linear(512, 3)
        else:
            self.decision = nn.Linear(512, 1)

    def forward(self, x):
        # print('00', x.shape, flush=True)
        for module_pos, module in self.features._modules.items():
            # 7
            # print('-',x.size(),flush=True)
            x = module(x)
            # print('{}, {}:\n{}\n'.format(module_pos, self.features._modules[module_pos], x.shape), flush=True)
        x = self.avgpool(x)
        # print('000', x.shape, flush=True)
        x = x.view(x.size(0), -1)
        # print('001', x.shape, flush=True)
        x = self.decision(x)
        # print('002', x.shape, flush=True)
        x = x.view(x.size(0), -1)
        # print('003', x.shape, flush=True)
        return x

    # def forward(self, x):
    #     x = self.features(x)
    #     x = self.avgpool(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.decision(x)
    #     x = x.view(x.size(0), -1)
    #     return x


class ModifiedVGG(nn.Module):
    def __init__(self, pretrained_model, learn_pos=False):
        super(ModifiedVGG, self).__init__()
        self.features = pretrained_model.features

        ## Original classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(True)
        )

        if learn_pos:
            self.output = nn.Linear(2048, 3)
        else:
            self.output = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x
