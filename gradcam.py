"""
Adapted from:

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import cv2
import numpy as np
import torch
import os
from utils.torch_utils import load_checkpoint, get_feature_model, get_optimizer, tensor_to_var, get_q_model, \
    var_to_numpy, tensor_to_numpy
from utils.misc_fcts_visualization import save_class_activation_on_image, one_img_figure, save_orig_with_bbs

from prepare_feat import get_valloader_only
import torchvision.models as models
import models as m
import torchvision.models as torch_m

import torch.nn as nn
import torch.optim as optim
import resnet as res


class CamExtractor(object):
    """
        Extracts cam features from the model
    """

    def __init__(self, model, target_layer, model_type, feat):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.model_type = model_type
        self.feat = feat

    def save_gradient(self, grad):
        self.gradients = grad

    # def forward_pass_on_conv_get_last_layer(self, x):
    #     conv_output = None
    #     for module_pos, module in self.model.features._modules.items():
    #         # print('\n- Upper level module: {} -'.format(module_pos))
    #         if isinstance(module, nn.Sequential):
    #             for _module_pos, _module in module._modules.items():
    #                 # print('- Middel level module: {} -'.format(_module_pos))
    #                 if isinstance(_module, torch_m.resnet.BasicBlock):
    #                     for __module_pos, __module in _module._modules.items():
    #                         if __module_pos == 'downsample':
    #                             pass
    #                         else:
    #                             x = __module(x)
    #                             # print('- Lower level module: {} -'.format(__module_pos))
    #                             if __module_pos == 'conv2' and _module_pos=='1' and module_pos == "7":
    #                                 x.register_hook(self.save_gradient)
    #                                 conv_output = x
    #         else:
    #             x = module(x)
    #
    #     return conv_output, x

    def forward_pass_on_conv_get_last_layer(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x.register_hook(self.save_gradient)
        conv_output = x
        return conv_output, x
    
#     def forward_pass_on_conv_get_last_layer(self, x):
#         conv_output = None
#         for module_pos, module in self.model.features._modules.items():
#             if int(module_pos) <= 6:
#                 x = module(x)
#             elif int(module_pos) == 7:
#                 x = module(x)
#                 x.register_hook(self.save_gradient)
#                 conv_output = x  # Save the convolution output on that layer
#         return conv_output, x

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        if self.model_type == 'auto':
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x.register_hook(self.save_gradient)
            conv_output = x
        else:
            for module_pos, module in self.model.features._modules.items():
                print('What module?', module_pos, module)
                print('What input?', x.shape)
                x = module(x)  # Forward
                if int(module_pos) == self.target_layer:
                    x.register_hook(self.save_gradient)
                    conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x, on_conv):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions

        if on_conv:
            # Forward pass on the classifier
            if self.model_type == 'vgg':
                output, x = self.forward_pass_on_convolutions(x)
                x = x.view(x.size(0), -1)
                x = self.model.classifier(x)
                x = x.view(x.size(0), -1)
                x = self.model.output(x)
            elif self.model_type == 'resnet':
                output, x = self.forward_pass_on_conv_get_last_layer(x)
                x = self.model.avgpool(x)
                x = x.view(x.size(0), -1)
#                 x = self.model.decision(x)
                x = self.model.fc(x)
                x = x.view(x.size(0), -1)
            elif self.model_type == 'simple':
                output, x = self.forward_pass_on_convolutions(x)
                x = x.view(x.size(0), -1)
                if self.feat:
                    x = self.model.l1(x)
                    x = self.model.decision(x)
                    x = x.view(x.size(0), -1)
                else:
                    x = self.model.qnet(x)
                    x = x.view(x.size(0), -1)
            elif self.model_type == 'auto':
                print('-0',x.shape)
                output, x = self.forward_pass_on_convolutions(x)
                x = x.view(x.size(0), -1)
                if self.feat:
                    raise ValueError('No gradcam for auto features please')
                else:
                    x = self.model.qnet(x)
                    x = x.view(x.size(0), -1)
        else:
            x = self.model.features(x)
            if self.model_type == 'vgg':
                x = self.model.classifier(x)
                x.register_hook(self.save_gradient)
                output = x
                x = x.view(x.size(0), -1)
                x = self.model.output(x)
            elif self.model_type == 'resnet':
                x = self.model.avgpool(x)
                x.register_hook(self.save_gradient)
                output = x
                x = x.view(x.size(0), -1)
                x = self.model.decision(x)
                x = x.view(x.size(0), -1)
            elif self.model_type == 'simple':
                x = x.view(x.size(0), -1)
                x = self.model.l1(x)
                x.register_hook(self.save_gradient)
                output = x
                x = self.model.decision(x)
                x = x.view(x.size(0), -1)
            elif self.model_type == 'auto':
                raise ValueError('No gradcam for auto features please')
        return output, x


class GradCam(object):
    """
        Produces class activation map
    """

    def __init__(self, model, target_layer, model_type, feat):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer, model_type, feat)
        self.model_type = model_type
        self.feat = feat

    def generate_cam(self, input_image, label=1, on_conv=True, pos=False, row=None, col=None):
        # For feature network label = 0,1 for qnet label = which action
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)

        output, model_output = self.extractor.forward_pass(tensor_to_var(input_image, force_on_cpu=False), on_conv)

        # Instead of one hot vector create say target is 1
        # if target_class is None:
        #     target_class = np.argmax(model_output.data.numpy())
        # # Target for backprop
        # one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        # one_hot_output[0][target_class] = 1

        # Zero grads
#         if self.model_type == 'auto':
        self.model.conv1.zero_grad()
        self.model.bn1.zero_grad()
        self.model.relu.zero_grad()
        self.model.maxpool.zero_grad()

        self.model.layer1.zero_grad()
        self.model.layer2.zero_grad()
        self.model.layer3.zero_grad()
        self.model.layer4.zero_grad()

#         else:
#             self.model.features.zero_grad()

        if self.feat:
            if self.model_type == 'vgg':
                self.model.classifier.zero_grad()
            elif self.model_type == 'resnet':
                self.model.avgpool.zero_grad()
#                 self.model.decision.zero_grad()
                self.model.fc.zero_grad()
            elif self.model_type == 'simple':
                self.model.l1.zero_grad()
                self.model.decision.zero_grad()

        # Backward pass with specified target
        if pos:
            tar = torch.FloatTensor(1, 3).zero_()
            tar[0][0] = label
            tar[0][1] = row
            tar[0][2] = col
        elif self.feat:
            tar = torch.FloatTensor(1, 5).zero_()
            tar[0][label] = label
        else:
            tar = torch.FloatTensor(1, 7).zero_()
            tar[0][label] = 10

        if torch.cuda.is_available():
            tar = tar.cuda()

        model_output.backward(gradient=tar, retain_graph=True)
        # Get hooked gradients
        guided_gradients = var_to_numpy(self.extractor.gradients)[0]
        # Get convolution outputs
        target = var_to_numpy(output)[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        print('01', cam.shape)
        cam = cv2.resize(cam, (224, 224))
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        return cam


def main():    
    # Last layer of convs is 28, maxpool is 30
    target_layer = 4
    on_conv = True
    
    no_imgs = 10
    toy = False
    feat = True
    label = 3 # 0,1 for feat, 0,6 for actions of qnet, 0,5 for cats
    if not feat:
        combi = True
        # inputsize = 2704
        # hiddensize = 512
        inputsize = 25088
        hiddensize = 1024
        outputsize = 7
        hidden_rec = 0
    else:
        combi = False
    recurrent = -1
    model_type = 'resnet'
    with_pos = False
    # checkpoint_dir_feat = os.path.abspath('/Users/lisa/Documents/Uni/ThesisDS/local_python/checkpoints/resnet_toy_01')

    # checkpoint_dir_feat = os.path.abspath('../checkpoints/res_pos_01')
    # checkpoint_filename_feat = os.path.join(checkpoint_dir_feat, 'checkpoint_resnet.pth.tar')
    checkpoint_dir_feat = os.path.abspath('../checkpoints_from_18_07_18/resnet_cats_01')
    checkpoint_filename_feat = os.path.join(checkpoint_dir_feat, 'warmup_model_{}.pth.tar'.format(model_type))
    # checkpoint_filename_feat = os.path.join(checkpoint_dir_feat, 'checkpoint_vgg.pth.tar')
    # checkpoint_filename_feat = os.path.join(checkpoint_dir_feat, 'model_best_resnet.pth.tar')
    # vgg = models.vgg16(pretrained=True)

    # checkpoint_dir_q = os.path.abspath('../checkpoints_from_18_07_18/qnet_simple_05_1')
    # checkpoint_filename_q = os.path.join(checkpoint_dir_q, 'model_best_qnet.pth.tar')

#     checkpoint_dir_feat = os.path.abspath('../checkpoints_from_18_07_18/resnet_auto_test_adam_1_2')
#     checkpoint_filename_feat = os.path.join(checkpoint_dir_feat, 'warmup_model_{}.pth.tar'.format(model_type))

#     checkpoint_dir_q = os.path.abspath('../checkpoints_from_18_07_18/qnet_auto_03')
#     checkpoint_filename_q = os.path.join(checkpoint_dir_q, 'model_best_qnet.pth.tar')

    lr = 0.01
    model = get_feature_model(model_type, model_type, load_pretrained=False, opti='optim.Adam', lr=lr, mom=None,
                              checkpoint_pretrained=None,
                              learn_pos=False, force_on_cpu=False,cat=True)


    if torch.cuda.is_available():
        model.cuda()

    print(model)
    if not feat:
        if model_type == 'auto' or model_type == 'resnet':
            model = res.ResNetFeatures(model)
        else:
            model = m.NetNoDecisionLayer(model)

        model = get_q_model(combi=combi, recurrent=recurrent, toy=toy, inputsize=inputsize, hiddensize=hiddensize, outputsize=outputsize, feature_model=model,
                            hidden_rec=hidden_rec)
    print(model)

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.SmoothL1Loss()
    if combi and recurrent <= 0:
        if model_type == 'auto' or model_type == 'resnet':
            model_params = [
                {'params': model.conv1.parameters(), 'lr': lr / 10},
                {'params': model.bn1.parameters(), 'lr': lr / 10},
                {'params': model.relu.parameters(), 'lr': lr / 10},
                {'params': model.maxpool.parameters(), 'lr': lr / 10},
                {'params': model.layer1.parameters(), 'lr': lr / 10},
                {'params': model.layer2.parameters(), 'lr': lr / 10},
                {'params': model.layer3.parameters(), 'lr': lr / 10},
                {'params': model.layer4.parameters(), 'lr': lr / 10},
                {'params': model.qnet.parameters()}
            ]
        else:
            model_params = [
                {'params': model.features.parameters(), 'lr': lr / 10},
                {'params': model.qnet.parameters()}
            ]
    else:
        model_params = model.parameters()
    optimizer = get_optimizer(model_params, 'optim.Adam', 0.01, None)

    if feat:
        to_load = os.path.join(checkpoint_dir_feat, checkpoint_filename_feat)
    else:
        to_load = os.path.join(checkpoint_dir_q, checkpoint_filename_q)

    if os.path.exists(to_load):
        model, _, _ = load_checkpoint(model, optimizer, filename=to_load, force_on_cpu=False)
    else:
        raise ValueError('Checkpoint {} does not exist.'.format(to_load))

    val_loader = get_valloader_only(toy=toy, rsyncing=False, batch_size=1, num_workers=0,cat=True)

    # Grad cam


    # Target layer doesn't matter for classifier hook
    # target_layer = 0
    # on_conv = False

    grad_cam = GradCam(model, target_layer=target_layer, model_type=model_type, feat=feat)
    checkpoint_dir_feat = os.path.join(checkpoint_dir_feat, 'Layer_{}'.format(target_layer))
    if not os.path.isdir(checkpoint_dir_feat):
        os.makedirs(checkpoint_dir_feat)
    # Generate cam mask

    for i, batch in enumerate(val_loader):
        if len(batch['image_batch']) > 1:
            for j in range(len(batch['image_batch'])):
                print('---', flush=True)
                print(
                    '{}_{}, label {}, has lesion {}'.format(batch['image_id_batch'][j], j, batch['label_batch'][j],
                                                            batch['has_lesion_batch'][j]), flush=True)
                print(batch['image_batch'][j].size(), flush=True)
                print(batch['image_batch'][j].type(), flush=True)
                # cam = grad_cam.generate_cam(torch.unsqueeze(batch['image_batch'][j], 0),
                #                             label=batch['label_batch'][j], on_conv=on_conv, pos=with_pos,
                #                             row=batch['center_row_batch'][j], col=batch['center_col_batch'][j])

                # cam = grad_cam.generate_cam(torch.unsqueeze(batch['image_batch'][j], 0),
                #                             label=1, on_conv=on_conv, pos=with_pos,
                #                             row=batch['center_row_batch'][j], col=batch['center_col_batch'][j])
                # save_class_activation_on_image(checkpoint_dir_feat, batch['image_batch'][j].numpy(), cam,
                #                                '{}_{}'.format(batch['image_id_batch'][j], j))

                cam = grad_cam.generate_cam(torch.unsqueeze(batch['image_batch'][j], 0),
                                            label=label, on_conv=on_conv, pos=with_pos,
                                            row=batch['center_row_batch'][j], col=batch['center_col_batch'][j])
                save_class_activation_on_image(checkpoint_dir_feat, tensor_to_numpy(batch['image_batch'][j]), cam,
                                               '{}_{}'.format(batch['image_id_batch'][j], j))
            one_img_figure(checkpoint_dir_feat, batch['image_id_batch'][0], batch['original_batch'][0], four_bbs=True)
            save_orig_with_bbs(checkpoint_dir_feat, batch['image_id_batch'][0], batch['original_batch'][0],
                               batch['bbox_batch'])
        elif len(batch['image_batch']) == 1:
            print('---', flush=True)
            print(batch['image_batch'][0].shape, flush=True)
            print('{}_{}, label {}, has lesion {}'.format(batch['image_id_batch'][0], 0, batch['label_batch'][0],
                                                          batch['has_lesion_batch'][0]), flush=True)
            # cam = grad_cam.generate_cam(torch.unsqueeze(batch['image_batch'][0], 0), label=batch['label_batch'][0],
            #                             on_conv=on_conv, pos=with_pos, row=batch['center_row_batch'][0],
            #                             col=batch['center_col_batch'][0])
            cam = grad_cam.generate_cam(torch.unsqueeze(batch['image_batch'][0], 0), label=label,
                                        on_conv=on_conv, pos=with_pos, row=batch['center_row_batch'][0],
                                        col=batch['center_col_batch'][0])
            save_class_activation_on_image(checkpoint_dir_feat,tensor_to_numpy(batch['image_batch'][0]), cam,
                                           '{}_{}'.format(batch['image_id_batch'][0], 0))

        if i >= no_imgs - 1:
            break
    print('Grad cam completed', flush=True)


if __name__ == '__main__': main()
