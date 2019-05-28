import os
import copy
import cv2
import numpy as np
import pdb
import torch
from torch.autograd import Variable
from torchvision import models

from utils.utils import get_nums_from_bbox
import matplotlib
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Palatino']
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from utils.dql_utils import string_action, string_case
from utils.torch_utils import tensor_to_numpy, var_to_numpy


def get_img_imshow(img):
    img = np.reshape(img, (3, img.shape[1], img.shape[2]))
    img = np.swapaxes(img, 0, 1)
    img = np.swapaxes(img, 1, 2)
    return img


def save_image_with_q_star(filename, q_star, row_lim, col_lim, spacing, original_img):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    width = original_img.shape[1]
    heigth = original_img.shape[0]

    X = np.arange(0, row_lim, spacing)
    Y = np.arange(0, col_lim, spacing)
    print('X', X.shape, flush=True)
    print('Y', Y.shape, flush=True)
    X, Y = np.meshgrid(X, Y)
    print('X', X.shape, flush=True)
    print('Y', Y.shape, flush=True)
    print('Q*', q_star.shape, flush=True)
    # surf = ax.plot_surface(X, Y, q_star, rstride=1, cstride=1, cmap=cm.winter,linewidth=0, antialiased=True)

    # ax.set_zlim(-2.01, 1.01)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # fn = get_sample_data("./lena.png", asfileobj=False)
    # arr = read_png(fn)

    arr = original_img

    # 10 is equal length of x and y axises of your surface
    stepX, stepY = 10. / arr.shape[0], 10. / arr.shape[1]

    X1 = np.arange(0, row_lim, stepX)
    Y1 = np.arange(0, col_lim, stepY)
    X1, Y1 = np.meshgrid(X1, Y1)
    # stride args allows to determine image quality 
    # stride = 1 work slow
    ax.plot_surface(X1, Y1, np.min(q_star) - 2, rstride=1, cstride=1, facecolors=arr)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def save_class_activation_on_image(checkpoint_dir, org_img, activation_map, file_name):
    '''
    Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    '''

    # Grayscale activation map

    print('Orig img', org_img.shape, type(org_img))
    print('acti map', activation_map.shape, type(activation_map))

    path_to_file = os.path.join(checkpoint_dir, file_name + '_Cam_Grayscale.jpg')
    cv2.imwrite(path_to_file, activation_map)
    # Heatmap of activation map
    activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_HSV)
    path_to_file = os.path.join(checkpoint_dir, file_name + '_Cam_Heatmap.jpg')
    cv2.imwrite(path_to_file, activation_heatmap)

    # Heatmap on picture
    # org_img = cv2.resize(org_img[0], (224, 224))
    org_img = np.swapaxes(org_img, 0, 1)
    org_img = np.swapaxes(org_img, 1, 2)

    if org_img.max() > 1:
        pass
    else:
        org_img = org_img * 255

    path_to_file = os.path.join(checkpoint_dir, file_name + '_Orig_Image.jpg')
    cv2.imwrite(path_to_file, org_img)

    alpha = 0.2
    img_heatmap_overlay = cv2.addWeighted(np.float32(activation_heatmap), alpha, np.float32(org_img), 1 - alpha, 0)
    path_to_file = os.path.join(checkpoint_dir, file_name + '_Cam_Overlay_Image.jpg')
    cv2.imwrite(path_to_file, img_heatmap_overlay)

    img_with_heatmap = np.float32(activation_heatmap) + np.float32(org_img)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    path_to_file = os.path.join(checkpoint_dir, file_name + '_Cam_On_Image.jpg')
    cv2.imwrite(path_to_file, np.uint8(255 * img_with_heatmap))


def one_img_figure(checkpoint_dir, img_id, big_org_img, four_bbs=False):
    if four_bbs:
        lim = 18
    else:
        lim = 3
    for i in range(lim):
        path_to_file = os.path.join(checkpoint_dir, '{}_{}_Orig_Image.jpg'.format(img_id, i))
        img_top = cv2.imread(path_to_file)
        path_to_file = os.path.join(checkpoint_dir, '{}_{}_Cam_Overlay_Image.jpg'.format(img_id, i))
        img_bottom = cv2.imread(path_to_file)
        # print(img_top.shape)
        # print(img_bottom.shape)
        current_img = cv2.vconcat((img_top, img_bottom))
        # print(current_img.shape)
        if i == 0:
            all_imgs = current_img.copy()
        else:
            all_imgs = cv2.hconcat((all_imgs, current_img))
    path_to_file = os.path.join(checkpoint_dir, '{}_all.jpg'.format(img_id))
    cv2.imwrite(path_to_file, all_imgs)


def save_orig_with_bbs(checkpoint_dir, img_id, big_org_img, bboxes):
    # big_org_img = np.swapaxes(big_org_img, 0, 1)
    # big_org_img = np.swapaxes(big_org_img, 1, 2)
    # big_org_img = np.float32(big_org_img *255)
    # path_to_file = os.path.join(checkpoint_dir, '{}_orig.jpg'.format(img_id))
    # cv2.imwrite(path_to_file, full)

    big_org_img = np.swapaxes(big_org_img, 0, 1)
    big_org_img = np.swapaxes(big_org_img, 1, 2)

    sizes = np.shape(big_org_img)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(big_org_img)
    for bb in bboxes:
        if bb:
            r, c, rn, cn = get_nums_from_bbox(bb)
            # Create a Rectangle patch
            edgecol = 'g'
            rect = patches.Rectangle((c, r), cn, rn, linewidth=0.25, edgecolor=edgecol, facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
    plt.savefig(os.path.join(checkpoint_dir, '{}_orig.jpg'.format(img_id)), dpi=height)
    plt.close()


def save_image_with_orig_plus_current_bb(input_image, filename, bbox_flag=False, r=0, c=0, rn=0, cn=0, ro=0, co=0, rno=0,
                                         cno=0, cm='binary', lwidth=0.5, Q_s=None, eps=-1, action=-1, case=None):
    # https://fengl.org/2014/07/09/matplotlib-savefig-without-borderframe/

    # sizes = np.shape(data)
    # height = float(sizes[0])
    # width = float(sizes[1])

    # fig = plt.figure()
    # fig.set_size_inches(width/height, 1, forward=False)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)

    # print('Original BBox is: row {} col {} row_no {} col_no {}'.format(ro,co,rno,cno))
    # print('Current BBox is: row {} col {} row_no {} col_no {}'.format(r, c, rn, cn))
    image = np.copy(input_image)
#     print(np.min(image),np.mean(image),np.max(image),flush=True)
    if np.max(image)>1:
#         print('clip to 0 255',flush=True)
        image = np.clip(image, 0, 255)
        image = image.astype(int)
    else:
#         print('clip to 0 1',flush=True)
        image = np.clip(image, 0.0, 1.0)
    fig, ax = plt.subplots(1, 1)
    ax.set_axis_off()
#     print(np.min(image),np.mean(image),np.max(image),flush=True)
    ax.imshow(get_img_imshow(image), cmap=cm)
    if bbox_flag:
        # Create a Rectangle patch
        edgecol = 'g'
        rect = patches.Rectangle((co, ro), cno, rno, linewidth=lwidth, edgecolor=edgecol, facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

        edgecol = 'r'
        rect = patches.Rectangle((c, r), cn, rn, linewidth=lwidth, edgecolor=edgecol, facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
#     from trainer_qnet import QNetTrainer
    # a = QNetTrainer.get_best_action(Q_s)
    # print(Q_s)
    # print(Q_s.max())
    # print(np.amax(Q_s))
    ma = np.amax(Q_s)
    Q_s = np.squeeze(Q_s)

    if eps > -1:
        # print(Q_s.shape)
        string = 'Down: {:.2f}'.format(Q_s[0])
        string = 'Whatever {:.2f}'.format(ma)
        string = 'Jajaja {}'.format(string_action(action))
        string = 'Finally {:.2f}'.format(eps)
        string = 'Case {}'.format(string_case(case))
        title_string = 'Down: {:.2f}, Up: {:.2f}, Right: {:.2f}, Left: {:.2f}\nBigger: {:.2f}, Smaller: {:.2f}, Trigger: {:.2f}\nMax(Q): {:.2f}, Action: {}\nEpsilon: {:.2f}, Case: {}'.format(
            Q_s[0], Q_s[1], Q_s[2], Q_s[3], Q_s[4], Q_s[5], Q_s[6], ma, string_action(action), eps,
            string_case(case))
    else:
        string = 'Down: {:.2f}'.format(Q_s[0])
        string = 'Whatever {:.2f}'.format(ma)
        string = 'Jajaja {}'.format(string_action(action))
        title_string = 'Down: {:.2f}, Up: {:.2f}, Right: {:.2f}, Left: {:.2f}\nBigger: {:.2f}, Smaller: {:.2f}, Trigger: {:.2f}\nMax(Q): {:.2f}, Action: {}'.format(
            Q_s[0], Q_s[1], Q_s[2], Q_s[3], Q_s[4], Q_s[5], Q_s[6], ma, string_action(action))
    plt.title(title_string)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
