import os
from prepare_feat import get_dataloaders, get_valloader_only

# train_loader, val_loader = get_dataloaders('test',rsyncing=False,selective_sampling=False,warmup_trainer=None,batch_size=16,num_workers=os.cpu_count()-1,seed=12345,data_aug_vec=[0.5,0.25,0.5,0.5])

val_loader = get_valloader_only(False, rsyncing=False, batch_size=16, num_workers=os.cpu_count() - 1, seed=12345)
