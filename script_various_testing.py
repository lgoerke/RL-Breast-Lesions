import os
import torch
import torchvision
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler

from environment import MammoEnv
import utils.datasets as u
from prepare_feat import get_valloader_only
from utils.torch_utils import get_feature_model, list_to_variable, tensor_to_var
import utils.transforms as t
from utils.utils import get_paths
import time
import json
import models
import numpy as np


def change_width_height_bbox():
    print('Get paths')
    anno_path_train, anno_path_val, _ = get_paths(False, toy=True)
    print('Open json')
    # with open(anno_path_val, "r") as jsonFile:
    #     data = json.load(jsonFile)
    #
    # print('Get Categories')
    # new_annos = []
    # for entry in data['annotations']:
    #     # print(entry)
    #     c,r,rn,cn = entry['bbox']
    #     entry['bbox'] = c,r,cn,rn
    #     new_annos.append(entry)
    # data['annotations'] = []
    # data['annotations'] = new_annos
    #
    # with open(anno_path_val, "w") as jsonFile:
    #     json.dump(data, jsonFile)


def test_screenpoint3():
    checkpoint_dir = '../../checkpoints_from_18_07_18/qnet_23'
    print(os.path.abspath(checkpoint_dir))
    rsyncing = False
    batch_size = 1
    multiprocessing = False
    # num_workers=os.cpu_count() - 1
    num_workers = 0
    data_aug_vec = [0.5, 0.25, 0.5, 0.5]
    toy = False
    notebook = False

    anno_path_train, anno_path_val, png_path = get_paths(rsyncing, toy, notebook)

    # TODO
    # png_path = os.path.join('/Users/lisa/Documents/Uni/ThesisDS/thesis_ds/one_img_dataset', 'png')
    # anno_path_train = os.path.join('/Users/lisa/Documents/Uni/ThesisDS/thesis_ds/one_img_dataset',
    #                                'annotations/mscoco_train_full.json')
    # anno_path_val = os.path.join('/Users/lisa/Documents/Uni/ThesisDS/thesis_ds/one_img_dataset',
    #                                'annotations/mscoco_train_full.json')

    print('Creating Coco Datasets..', flush=True)
    # t.ToTensor()
    trans_img = torchvision.transforms.Compose([t.BboxCrop(targetsize=224), t.RandomFlipImg(prob=data_aug_vec[0]),
                                                t.RandomGammaImg(prob=data_aug_vec[1], use_normal_distribution=True)])
    trans_bb = torchvision.transforms.Compose(
        [t.GetFourBBs(), t.RandomTranslateBB(prob=data_aug_vec[2], pixel_range=10),
         t.RandomScaleBB(prob=data_aug_vec[3], max_percentage=0.1)])

    trainset = u.dataset_coco(png_path, anno_path_train, transform=trans_img, bbox_transform=trans_bb)
    print('Training set has', len(trainset), 'images', flush=True)

    valset = u.dataset_coco(png_path, anno_path_val,
                            transform=torchvision.transforms.Compose([t.GetFourBBs(), t.BboxCrop(targetsize=224)]))
    print('Validation set has', len(valset), 'images', flush=True)

    print('Len trainset', len(trainset))
    print('Len valset', len(valset))
    print('Example set', trainset[3])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=SequentialSampler(trainset),
                                              num_workers=num_workers, collate_fn=u.mammo_collate,
                                              pin_memory=multiprocessing)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, sampler=SequentialSampler(valset),
                                            num_workers=num_workers, collate_fn=u.mammo_collate,
                                            pin_memory=multiprocessing)

    print('Len trainloader', len(trainloader))
    print('Len valloader', len(valloader))
    empty_batches = 0
    filled_batches = 0
    for idx, batch in enumerate(trainloader):
        # print('----')
        # print('{} th batch of {}'.format(idx,len(trainloader)),flush=True)
        # if len(batch['image_batch']) == 0:
        # 	empty_batches += 1
        # else:
        # 	filled_batches += 1
        # print(batch)
        # break
        pass

    # print('Empty batches',empty_batches)
    # print('Filled batches', filled_batches)

    # print('Getting weights for sampling..', flush=True)
    # weights = prepare_feat.get_weights(checkpoint_dir, trainloader, training=True)

    # print('Len weights',len(weights))


def test_smaller_bigger():
    model = get_feature_model('resnet', 'resnet')

    max_num_imgs_val = 10

    zeta = 1
    eta = 10

    tau = 0.6

    print('Getting path ready..', flush=True)
    start_time = time.time()
    print('Creating Coco Datasets..', flush=True)

    png_path = os.path.join('/Volumes/breast/projects/lisa/koel/one_img_dataset', 'png')
    anno_path_train = os.path.join('/Volumes/breast/projects/lisa/koel/one_img_dataset',
                                   'annotations/mscoco_train_full.json')

    print(os.path.abspath(anno_path_train), flush=True)

    trainset = u.dataset_coco(png_path, anno_path_train)
    print('Training set has', len(trainset), 'images', flush=True)
    val_env = MammoEnv(trainset, eta=eta, zeta=zeta, tau=tau, model=model, one_img=True, max_no_imgs=max_num_imgs_val)
    total_time = time.time() - start_time
    print('Creating Datasets took {:.0f} seconds.'.format(total_time), flush=True)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    val_env.reset()
    dsc = val_env.get_current_dice()
    img = val_env.render(mode='human', with_state=True)
    ax.imshow(img)
    ax.set_title('{}'.format(dsc))
    fig.canvas.draw()
    val_env.step(3)
    dsc = val_env.get_current_dice()
    img = val_env.render(mode='human', with_state=True)
    ax.imshow(img)
    ax.set_title('{}'.format(dsc))
    fig.canvas.draw()
    val_env.step(5)
    dsc = val_env.get_current_dice()
    img = val_env.render(mode='human', with_state=True)
    ax.imshow(img)
    ax.set_title('{}'.format(dsc))
    fig.canvas.draw()
    val_env.step(5)
    dsc = val_env.get_current_dice()
    img = val_env.render(mode='human', with_state=True)
    ax.imshow(img)
    ax.set_title('{}'.format(dsc))
    fig.canvas.draw()
    val_env.step(5)
    dsc = val_env.get_current_dice()
    img = val_env.render(mode='human', with_state=True)
    ax.imshow(img)
    ax.set_title('{}'.format(dsc))
    fig.canvas.draw()

    # val_env.step(3)
    # dsc = val_env.get_current_dice()
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # ax.set_title('{}'.format(dsc))
    # fig.canvas.draw()
    # val_env.step(5)
    # dsc = val_env.get_current_dice()
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # ax.set_title('{}'.format(dsc))
    # fig.canvas.draw()
    # val_env.step(3)
    # dsc = val_env.get_current_dice()
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # ax.set_title('{}'.format(dsc))
    # fig.canvas.draw()
    # val_env.step(5)
    # dsc = val_env.get_current_dice()
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # ax.set_title('{}'.format(dsc))
    # fig.canvas.draw()
    # val_env.step(3)
    # dsc = val_env.get_current_dice()
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # ax.set_title('{}'.format(dsc))
    # fig.canvas.draw()
    # val_env.step(1)
    # dsc = val_env.get_current_dice()
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # ax.set_title('{}'.format(dsc))
    # fig.canvas.draw()
    # val_env.step(5)
    # dsc = val_env.get_current_dice()
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # ax.set_title('{}'.format(dsc))
    # fig.canvas.draw()
    # val_env.step(5)
    # dsc = val_env.get_current_dice()
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # ax.set_title('{}'.format(dsc))
    # fig.canvas.draw()
    # val_env.step(5)
    # dsc = val_env.get_current_dice()
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # ax.set_title('{}'.format(dsc))
    # fig.canvas.draw()
    # val_env.step(5)
    # dsc = val_env.get_current_dice()
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # ax.set_title('{}'.format(dsc))
    # fig.canvas.draw()
    # val_env.step(5)
    # dsc = val_env.get_current_dice()
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # ax.set_title('{}'.format(dsc))
    # fig.canvas.draw()
    # val_env.step(2)
    # dsc = val_env.get_current_dice()
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # ax.set_title('{}'.format(dsc))
    # fig.canvas.draw()
    # val_env.step(5)
    # dsc = val_env.get_current_dice()
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # ax.set_title('{}'.format(dsc))
    # fig.canvas.draw()

    # val_env.reset()
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # fig.canvas.draw()
    # val_env.step(4)
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # fig.canvas.draw()
    # val_env.step(4)
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # fig.canvas.draw()
    # val_env.step(4)
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # fig.canvas.draw()
    # val_env.step(4)
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # fig.canvas.draw()
    # val_env.step(4)
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # fig.canvas.draw()
    # val_env.step(4)
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # fig.canvas.draw()
    # val_env.step(4)
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # fig.canvas.draw()
    # val_env.step(4)
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # fig.canvas.draw()
    # val_env.step(4)
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # fig.canvas.draw()
    # val_env.step(4)
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # fig.canvas.draw()
    # val_env.step(4)
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # fig.canvas.draw()
    # val_env.step(4)
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # fig.canvas.draw()
    # val_env.step(4)
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # fig.canvas.draw()
    # val_env.step(4)
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # fig.canvas.draw()
    # val_env.step(4)
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # fig.canvas.draw()
    # val_env.step(4)
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # fig.canvas.draw()
    # val_env.step(4)
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # fig.canvas.draw()

    # val_env.reset()
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # fig.canvas.draw()
    # val_env.step(2)
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # fig.canvas.draw()
    # val_env.step(2)
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # fig.canvas.draw()
    # val_env.step(2)
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # fig.canvas.draw()
    # val_env.step(2)
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # fig.canvas.draw()
    # val_env.step(2)
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # fig.canvas.draw()
    # val_env.step(2)
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # fig.canvas.draw()
    # val_env.step(1)
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # fig.canvas.draw()
    # val_env.step(4)
    # img = val_env.render(mode='human')
    # ax.imshow(img)
    # fig.canvas.draw()

    # plt.show()

def test_resnet_auto():
    model = get_feature_model('auto', 'auto', load_pretrained=False, opti=None, lr=None, mom=None,
                      checkpoint_pretrained=None,
                      learn_pos=False, force_on_cpu=False)
    val_loader = get_valloader_only(False,False,1,0,False)
    for batch_idx, batch in enumerate(val_loader):
        if len(batch['image_batch']) > 0:
            # print(len(batch['image_batch']))
            # print(batch['image_batch'][0].size())
            data = list_to_variable(batch['image_batch'])
            labels = tensor_to_var(torch.Tensor(np.array(batch['label_batch'])), async=True)

            pred = model(data)

        break

def test_dataloader_iter():
    valloader = get_valloader_only(toy=True, rsyncing=False, batch_size=1, num_workers=0)
    valiter = iter(valloader)
    for i in range(100000):
        try:
            next(valiter)
        except StopIteration:
            print("YIPPIE {}".format(i))
            break
    print('Schluss mit der Schleife!')

if __name__ == '__main__':
    # test_dataloader_iter()
    # test_screenpoint3()
    # test_smaller_bigger()
    # print('Start')
    # change_width_height_bbox()
    test_resnet_auto()
