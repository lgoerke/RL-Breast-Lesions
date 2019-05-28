import torch

from torch.autograd import Variable
import torch.optim as optim
import torchvision.models as models

from torch import nn
import models as m

### Testing something about the dataset ###

# print('Getting path ready..',flush=True)
# root = "/mnt/synology/breast/archives/screenpoint1/processed_dataset"
# png_path = os.path.join(root,'png')
# anno_path_train = os.path.join(root, 'annotations/mscoco_train_full.json')
# anno_path_val = os.path.join(root, 'annotations/mscoco_val_full.json')

# print('Creating Coco Datasets..',flush=True)
# # TODO transform for downsampling
# trainset = u.dataset_coco(png_path,anno_path_train)
# print('Training set has',len(trainset),'images')

# idx = -1
# for i,img in enumerate(tqdm(trainset)):
# 	try:
# 		if img['image_id'] == 933:
# 			idx = i
# 	except TypeError:
# 		print(img)

# print(idx)
# print(trainset[idx]['image_height'])
# print(trainset[idx]['image_width'])
# print(trainset[idx]['bbox'])


### Testing how long it takes to create the dataset if it's only one img

# print('Getting path ready..',flush=True)
# start_time = time.time()
# print('Creating Coco Datasets..',flush=True)

# png_path = os.path.join('/mnt/synology/breast/projects/lisa/koel/one_img_dataset','png')
# anno_path_train = os.path.join('/mnt/synology/breast/projects/lisa/koel/one_img_dataset', 'annotations/mscoco_train_full.json')

# trainset = u.dataset_coco(png_path,anno_path_train)
# total_time = time.time() - start_time
# print('Creating Datasets took {:.0f} seconds.'.format(total_time))

# print('Training set has',len(trainset),'images')
# print(trainset[0])


### Testing how to print grad norm ###
resnet = models.resnet18(pretrained=True)
model = m.ModifiedResNet(resnet)

criterion = nn.BCEWithLogitsLoss()
feat_opti = optim.Adam(model.parameters())

input_img = torch.ones(3, 3, 224, 224).float()
preds = model(Variable(input_img))

targets = torch.zeros(3, 1).float()
targets = Variable(targets)

loss = criterion(preds, targets)

feat_opti.zero_grad()
loss.backward()

for p in model.parameters():
    print('===========\ngradient norm:{}'.format(torch.norm(p.grad)), flush=True)
