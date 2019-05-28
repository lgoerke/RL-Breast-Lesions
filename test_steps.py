import os
import numpy as np
import utils.utils as u
from prepare_feat import get_dataloaders
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def get_img_imshow(img):
    img = np.reshape(img, (3, img.shape[1], img.shape[2]))
    img = np.swapaxes(img, 0, 1)
    img = np.swapaxes(img, 1, 2)
    return img


def save_image_with_bb(index, batch, filename, bbox_flag=False, r=0, c=0, rn=0, cn=0, ro=0, co=0, rno=0, cno=0,
                       cm='binary'):
    # https://fengl.org/2014/07/09/matplotlib-savefig-without-borderframe/

    # sizes = np.shape(data)
    # height = float(sizes[0])
    # width = float(sizes[1])

    # fig = plt.figure()
    # fig.set_size_inches(width/height, 1, forward=False)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)

    fig, ax = plt.subplots(1, 2)
    ax[1].imshow(get_img_imshow(batch['image_batch'][index]), cmap=cm)
    ax[1].set_title('Label={}'.format(batch['label_batch'][index]))

    ax[0].imshow(get_img_imshow(batch['original_batch'][index]), cmap=cm)
    if bbox_flag:
        # Create a Rectangle patch
        edgecol = 'g'
        rect = patches.Rectangle((co, ro), cno, rno, linewidth=0.5, edgecolor=edgecol, facecolor='none')
        # Add the patch to the Axes
        ax[0].add_patch(rect)

        edgecol = 'r'
        rect = patches.Rectangle((c, r), cn, rn, linewidth=0.5, edgecolor=edgecol, facecolor='none')
        # Add the patch to the Axes
        ax[0].add_patch(rect)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def test_triple(no_imgs=10):
    checkpoint_dir = os.path.abspath('../checkpoints/test_steps_04')
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # print('------------------------')
    _, train_loader = get_dataloaders(checkpoint_dir, rsyncing=False, selective_sampling=False, warmup_trainer=None,
                                      batch_size=1, num_workers=0, data_aug_vec=[0, 0, 0, 0])
    # print('------------------------')
    cnt = 0
    for i, batch in enumerate(train_loader):
        # print('triple {}'.format(i))
        # print('Len batch', len(batch['has_lesion_batch']))
        if len(batch['image_batch']) > 1:
            for j in range(len(batch['image_batch'])):
                if batch['bbox_batch'][j]:
                    r, c, rn, cn = u.get_nums_from_bbox(batch['bbox_batch'][j])
                    ro, co, rno, cno = u.get_nums_from_bbox(batch['obbox_batch'][j])
                    bbox_flag = True
                else:
                    r = c = rn = cn = ro = co = rno = cno = 0
                    bbox_flag = False
                save_image_with_bb(j, batch, os.path.join(checkpoint_dir, 'triple_{}.png'.format(cnt)),
                                   bbox_flag=bbox_flag, r=r, c=c, rn=rn, cn=cn, ro=ro, co=co, rno=rno, cno=cno)
                cnt += 1

        elif len(batch['image_batch']) == 1:
            if batch['bbox_batch'][0]:
                r, c, rn, cn = u.get_nums_from_bbox(batch['bbox_batch'][0])
                ro, co, rno, cno = u.get_nums_from_bbox(batch['obbox_batch'][0])
                bbox_flag = True
            else:
                r = c = rn = cn = ro = co = rno = cno = 0
                bbox_flag = False
            save_image_with_bb(0, batch, os.path.join(checkpoint_dir, 'triple_{}.png'.format(cnt)), bbox_flag=bbox_flag,
                               r=r, c=c, rn=rn, cn=cn, ro=ro, co=co, rno=rno, cno=cno)
            cnt += 1

        # print('-+-+-+------------------')
        if i >= no_imgs:
            break


def test_data_augmentations(no_imgs=10):
    checkpoint_dir = os.path.abspath('../checkpoints/test_steps_04')
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # print('------------------------')
    train_loader, _ = get_dataloaders(checkpoint_dir, rsyncing=False, selective_sampling=False, warmup_trainer=None,
                                      batch_size=10, num_workers=0, data_aug_vec=[0.5, 0.25, 0.5, 0.5])
    # print('------------------------')
    for i, batch in enumerate(train_loader):
        print('Random {}'.format(i))
        # print('Len batch', len(batch['has_lesion_batch']))
        if batch['bbox_batch'][0]:
            r, c, rn, cn = u.get_nums_from_bbox(batch['bbox_batch'][0])
            ro, co, rno, cno = u.get_nums_from_bbox(batch['obbox_batch'][0])
            bbox_flag = True
        else:
            r = c = rn = cn = ro = co = rno = cno = 0
            bbox_flag = False
        save_image_with_bb(batch, os.path.join(checkpoint_dir, 'rnd_{}.png'.format(i)), bbox_flag=bbox_flag, r=r, c=c,
                           rn=rn, cn=cn, ro=ro, co=co, rno=rno, cno=cno)
        # print('-+-+-+------------------')
        if i >= no_imgs:
            break

    # train_loader, _ = get_dataloaders(checkpoint_dir,rsyncing=False,selective_sampling=False,warmup_trainer=None,batch_size=1,data_aug_vec=[1,0,0,0])
    # for i,batch in enumerate(train_loader):
    #     print('Flip {}'.format(i))
    #     if batch['bbox_batch'][0]:
    #         r,c,rn,cn = u.get_nums_from_bbox(batch['bbox_batch'][0])
    #         ro,co,rno,cno = u.get_nums_from_bbox(batch['obbox_batch'][0])
    #         bbox_flag = True
    #     else:
    #         r=c=rn=cn=ro=co=rno=cno=0
    #         bbox_flag=False
    #     save_image_with_bb(batch, os.path.join(checkpoint_dir,'flip_{}.png'.format(i)),bbox_flag =bbox_flag, r=r, c=c, rn=rn, cn=cn,ro=ro,co=co,rno=rno,cno=cno)
    #     if i >= no_imgs:
    #         break

    # train_loader, _ = get_dataloaders(checkpoint_dir,rsyncing=False,selective_sampling=False,warmup_trainer=None,batch_size=1,data_aug_vec=[0,1,0,0])
    # for i,batch in enumerate(train_loader):
    #     print('Gamma {}'.format(i))
    #     if batch['bbox_batch'][0]:
    #         r,c,rn,cn = u.get_nums_from_bbox(batch['bbox_batch'][0])
    #         ro,co,rno,cno = u.get_nums_from_bbox(batch['obbox_batch'][0])
    #         bbox_flag = True
    #     else:
    #         r=c=rn=cn=ro=co=rno=cno=0
    #         bbox_flag=False
    #     save_image_with_bb(batch, os.path.join(checkpoint_dir,'gamma_{}.png'.format(i)),bbox_flag =bbox_flag, r=r, c=c, rn=rn, cn=cn,ro=ro,co=co,rno=rno,cno=cno)
    #     if i >= no_imgs:
    #         break

    # train_loader, _ = get_dataloaders(checkpoint_dir,rsyncing=False,selective_sampling=False,warmup_trainer=None,batch_size=1,data_aug_vec=[0,0,1,0])
    # for i,batch in enumerate(train_loader):
    #     print('Translate {}'.format(i))
    #     if batch['bbox_batch'][0]:
    #         r,c,rn,cn = u.get_nums_from_bbox(batch['bbox_batch'][0])
    #         ro,co,rno,cno = u.get_nums_from_bbox(batch['obbox_batch'][0])
    #         bbox_flag = True
    #     else:
    #         r=c=rn=cn=ro=co=rno=cno=0
    #         bbox_flag=False
    #     save_image_with_bb(batch, os.path.join(checkpoint_dir,'translate_{}.png'.format(i)),bbox_flag =bbox_flag, r=r, c=c, rn=rn, cn=cn,ro=ro,co=co,rno=rno,cno=cno)
    #     if i >= no_imgs:
    #         break

    # train_loader, _ = get_dataloaders(checkpoint_dir,rsyncing=False,selective_sampling=False,warmup_trainer=None,batch_size=1,data_aug_vec=[0,0,0,1])
    # for i,batch in enumerate(train_loader):
    #     print('Scale {}'.format(i))
    #     if batch['bbox_batch'][0]:
    #         r,c,rn,cn = u.get_nums_from_bbox(batch['bbox_batch'][0])
    #         ro,co,rno,cno = u.get_nums_from_bbox(batch['obbox_batch'][0])
    #         bbox_flag = True
    #     else:
    #         r=c=rn=cn=ro=co=rno=cno=0
    #         bbox_flag=False
    #     save_image_with_bb(batch, os.path.join(checkpoint_dir,'scale_{}.png'.format(i)),bbox_flag =bbox_flag, r=r, c=c, rn=rn, cn=cn,ro=ro,co=co,rno=rno,cno=cno)
    #     if i >= no_imgs:
    #         break


test_triple()
