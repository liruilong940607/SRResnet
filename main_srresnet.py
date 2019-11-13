import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from srresnet import _NetG
from torchvision import models
import torch.utils.model_zoo as model_zoo

import glob
import random
import cv2
import numpy as np
import tqdm
import imageio

def To_tensor(img):
    return torch.from_numpy(img.transpose(2, 0, 1)).float()

def To_numpy(tensor):
    return tensor.detach().cpu().numpy().transpose(1, 2, 0)


def load_img(filename):
    return imageio.imread(filename, format='EXR-FI')

def save_img(filename_out, img, skip_if_exist=False):    
    if skip_if_exist and os.path.exists(filename_out):
        return
    os.makedirs(os.path.dirname(filename_out), exist_ok=True)
    imageio.imwrite(filename_out, img, format='EXR-FI')

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=200, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=4, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--vgg_loss", action="store_true", help="Use content loss?")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--dataset", type=str) # [spec, disph, displ, albedo, albedo256]

class Dataset():
    def __init__(self, dataset):
        if dataset == "albedo":
            self.HR_dir = "/home/ICT2000/rli/mnt/glab2/ForRuilong/home/workspace/SuperResolution/mmsr/datasets/albedo_train_sub/*.exr"
            # RGB
            self.mean = np.array([0.03996306, 0.07294353, 0.0922085])
            self.std = np.array([0.07891982, 0.08981047, 0.1882349])
        elif dataset == "spec":
            self.HR_dir = "/home/ICT2000/rli/mnt/glab2/ForRuilong/home/workspace/SuperResolution/mmsr/datasets/spec_train_sub/*.exr"
            # RGB
            self.mean = np.array([0.1474461, 0.1474461, 0.1474461])
            self.std = np.array([0.27835602, 0.27835602, 0.27835602])
        elif dataset == "disph":
            self.HR_dir = "/home/ICT2000/rli/mnt/vgldb1/LightStageFaceDB/Datasets/FaceEncoding/Displacement_Separation_Exr/high/*.exr"
            # RGB
            self.mean = np.array([-5.6566824e-07, -5.6566824e-07, -5.6566824e-07])
            self.std = np.array([0.00336712, 0.00336712, 0.00336712])
    
        self.dataset = dataset
        self.images = glob.glob(self.HR_dir)
        print (f"{dataset} total data:", len(self.images))
        
        self.cache_idx = 0
        self.cache = None            
        
        if self.dataset == "disph":
            self.load()
        
    def __len__(self):
        if self.dataset == "disph":
            return 200000
        return len(self.images)
    
    def load(self):
        del self.cache
        total_images = glob.glob(self.HR_dir)
        self.images = total_images[self.cache_idx : self.cache_idx + 200]
        self.cache = [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in tqdm.tqdm(self.images)]
        self.cache_idx += 200
        self.cache_idx = self.cache_idx if self.cache_idx < len(total_images) else 0
    
    def __getitem__(self, index):
        if self.dataset == "disph":
            index = index % 200
            hr_img = self.cache[index]
            x = random.randint(0, 4096-480)
            y = random.randint(0, 4096-480)
            hr_img = hr_img[x:x+480, y:y+480, :].copy()
        else:
            hr_img = cv2.imread(self.images[index], cv2.IMREAD_UNCHANGED)
        
        hr_img = (hr_img[:, :, ::-1] - self.mean) / self.std
        
        if random.random() < 0.5:
            hr_img = hr_img[:, ::-1, :]
        
        lr_img = cv2.resize(hr_img, (0,0), fx=0.25, fy=0.25)
        
        hr_img = np.array(hr_img).transpose(2, 0, 1)
        lr_img = np.array(lr_img).transpose(2, 0, 1)
        return torch.from_numpy(lr_img).float(), torch.from_numpy(hr_img).float()
        
def main():

    global opt, model, netContent
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = Dataset(opt.dataset)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, \
        batch_size=opt.batchSize, shuffle=True)

    if opt.vgg_loss:
        print('===> Loading VGG model')
        netVGG = models.vgg19()
        netVGG.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'))
        class _content_model(nn.Module):
            def __init__(self):
                super(_content_model, self).__init__()
                self.feature = nn.Sequential(*list(netVGG.features.children())[:-1])
                
            def forward(self, x):
                out = self.feature(x)
                return out

        netContent = _content_model()

    print("===> Building model")
    model = _NetG()
    criterion = nn.MSELoss(size_average=False)

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        if opt.vgg_loss:
            netContent = netContent.cuda() 

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch, opt.dataset)
        save_checkpoint(model, epoch, opt.dataset)
        
        if opt.dataset == "spech":
            train_set.load()
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, \
                batch_size=opt.batchSize, shuffle=True)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr 

def train(training_data_loader, optimizer, model, criterion, epoch, name):

    lr = adjust_learning_rate(optimizer, epoch-1)
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):

        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        if opt.vgg_loss:
            content_input = netContent(output)
            content_target = netContent(target)
            content_target = content_target.detach()
            content_loss = criterion(content_input, content_target)

        optimizer.zero_grad()

        if opt.vgg_loss:
            netContent.zero_grad()
            content_loss.backward(retain_graph=True)

        loss.backward()

        optimizer.step()

        if iteration%10 == 0:
            if opt.vgg_loss:
                print("===> Epoch[{}]({}/{}): Loss: {:.5} Content_loss {:.5}".format(epoch, iteration, len(training_data_loader), loss.item(), content_loss.item()))
            else:
                print("===> Epoch[{}]({}/{}): Loss: {:.5}".format(epoch, iteration, len(training_data_loader), loss.item()))
                
        if iteration%3000 == 0:
            save_checkpoint(model, epoch, name)
            save_output(output, input, target, name, iteration)

def save_output(output, input, target, name, iteration):
    if not os.path.exists("save/"):
        os.makedirs("save/")
    out_path = "save/" + "{}_model_iteration_{}_output.exr".format(name, iteration)
    save_img(out_path, To_numpy(output[0]), skip_if_exist=False)
    out_path = "save/" + "{}_model_iteration_{}_target.exr".format(name, iteration)
    save_img(out_path, To_numpy(target[0]), skip_if_exist=False)
    out_path = "save/" + "{}_model_iteration_{}_input.exr".format(name, iteration)
    save_img(out_path, To_numpy(input[0]), skip_if_exist=False)
    
            
def save_checkpoint(model, epoch, name):
    model_out_path = "checkpoint/" + "{}_model_epoch_{}.pth".format(name, epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()
