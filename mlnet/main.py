from email import parser
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np

from datasets import *
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--img_path',type=str,help='path to the images')
parser.add_argument('--map_path',type=str,help='path to the maps')
parser.add_argument('--log_path',type=str,help='path to store the log file')
parser.add_argument('--output_path',type=str,help='path to store the model')
parser.add_argument('--batch_size',type=int,default=10 ,help='batch size')
parser.add_argument('--shape_r',type=int,default=240,help='num of rows in img after transforms')
parser.add_argument('--shape_c',type=int,default=320,help='num of cols in img after transforms')
parser.add_argument('--shape_r_gt',type=int,default=30,help='output img size')
parser.add_argument('--shape_c_gt',type=int,default=40,help='output img size')
parser.add_argument('--last_freeze_layer',type=int,default=23,help='last freeze layer in vgg16')
parser.add_argument('--n_epoch',type=int,default=10,help='num of epoch to run')
parser.add_argument('--sample_interval',type=int,default=20,help='interval between image sampling')
args = parser.parse_args()

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs(args.log_path+'/'+timestamp,exist_ok=True)
os.makedirs(args.output_path+'/'+timestamp,exist_ok=True)

#--------------
# preprocess and load the dataset
#--------------

img_transforms = transforms.Compose([
    transforms.Resize(size=(args.shape_r,args.shape_c)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])# according to the pytorch docs
])

map_transforms = transforms.Compose([
    transforms.Resize(size=(args.shape_r_gt,args.shape_c_gt)),
    transforms.ToTensor(),
])


train_dataset = ImgDataset(args.img_path+'/train',args.map_path+'/train',img_transforms,map_transforms)

train_dataloader = DataLoader(
    train_dataset,
    batch_size = args.batch_size,
    shuffle=True,
    num_workers=8
)

def show_img(imgs,maps,outputs):
    def inverse_normalize(tensor, mean, std):# 避免normalize之后显示出来的图像很奇怪
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor
    n = len(imgs)
    fig,axes = plt.subplots(3,n)
    for i in range(0,n):
        axes[0,i].imshow(np.asarray(transforms.functional.to_pil_image(inverse_normalize(imgs[i],mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]).detach())))
        axes[0,i].set_xticks([]);axes[0,i].set_yticks([])
        axes[1,i].imshow(transforms.functional.to_pil_image(maps[i]),cmap='gray')
        axes[1,i].set_xticks([]);axes[1,i].set_yticks([])
        axes[2,i].imshow(transforms.functional.to_pil_image(outputs[i]),cmap='gray')
        axes[2,i].set_xticks([]);axes[2,i].set_yticks([])
    return fig

#--------
# prepare the model and criterion and optimizer
#--------

device = 'cuda' if torch.cuda.is_available() else 'cpu'

prior_size = ( int(args.shape_r_gt / 10) , int(args.shape_c_gt / 10) ) # keep the same as the paper, prior_size = output_size/10

model = MLNet(prior_size).to(device)

# freezing Layer
for i,param in enumerate(model.parameters()):
  if i < args.last_freeze_layer:
    param.requires_grad = False

criterion = ModMSELoss(args.shape_r_gt,args.shape_c_gt).cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3,weight_decay=0.0005,momentum=0.9,nesterov=True)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-4)

#------------
# train
#------------
batches_done = 0
writer = SummaryWriter(args.log_path+'/'+timestamp+'/runs')
for epoch in range(args.n_epoch):
    for batch in tqdm(train_dataloader,desc='[epoch: %s]' % epoch):
            optimizer.zero_grad()

            imgs = batch['img'].to(device)
            maps = batch['map'].to(device)

            outputs = model(imgs)
            loss = criterion(outputs,maps,model.prior.clone())# clone to avoid the calc in criterion flow to original variables in the grad map
            loss.backward()
            optimizer.step()
            writer.add_scalars('loss',{'loss':loss},global_step=batches_done)
            if batches_done == 0:
                writer.add_graph(model,imgs)
            if batches_done % args.sample_interval ==0:
                writer.add_figure('figures',show_img(imgs,maps,outputs),global_step=batches_done)
            batches_done+=1
    writer.flush()

# Save model
torch.save(model.state_dict(), "%s/%s/epoch_%s.pth" % (args.output_path,timestamp, epoch))