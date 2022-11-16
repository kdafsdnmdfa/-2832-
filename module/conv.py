import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import matplotlib.pyplot as plt
import os
from utils.tsne import tsne
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict

# from utils.check_un_utils import evaluate_vit_distance
import numpy as np
from sklearn.metrics import accuracy_score,roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.metric import AverageMeter
from tqdm import tqdm
import pickle

__all__ = ['iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200']
using_ckpt = False

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.downsample = downsample
        self.stride = stride

    def forward_impl(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out        

    def forward(self, x):
        if self.training and using_ckpt:
            return checkpoint(self.forward_impl, x)
        else:
            return self.forward_impl(x)


class IResNet(nn.Module):
    # fc_scale = 7 * 7
    fc_scale = 4 * 4
    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(IResNet, self).__init__()
        self.extra_gflops = 0.0
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        return x


def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model


def iresnet18(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained,
                    progress, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained,
                    progress, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained,
                    progress, **kwargs)


def iresnet200(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet200', IBasicBlock, [6, 26, 60, 6], pretrained,
                    progress, **kwargs)


from .vit import CosFace
class iresnet100_cosface(nn.Module):
    def __init__(self,dim,num_class,GPU_ID):
        super().__init__()
        self.GPU_ID = GPU_ID
        self.net = iresnet100(pretrained=False, num_features=dim)
        self.loss = CosFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
        
    def forward(self, x, label= None):
        emb = self.net(x)
        if label is not None:
            x = self.loss(emb, label)
            return x, emb
        else:
            return emb
        
class iresnet100_cosface_(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.net = iresnet100(pretrained=False, num_features=dim)
        # self.loss = CosFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
        
    def forward(self, x, label= None):
        emb = self.net(x)
        return emb

    def get_feature(self, x):
        return self.net(x)

    
class iresnet100_metric(nn.Module):
    def __init__(self,dim):
        super(iresnet100_metric, self).__init__()
        self.net = iresnet100(pretrained=False, num_features=dim)
        checkpoint = torch.load('./results/un_ms1m/conv_un/conv_un_epoch0.pth')
        new_state_dict = OrderedDict()
        for k, v in checkpoint['arch'].items():
            name = k[4:]
            new_state_dict[name] = v

        self.net.load_state_dict(new_state_dict)
        
        
        self.freeze(self.net)
        self.metric = nn.Linear(128, 64)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)


    def freeze(self, feat):
        for name, child in feat.named_children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x1, x2,sim = True):
        fea1 = self.net(x1)
        fea2 = self.net(x2)
        fea1 = self.metric(fea1)
        fea2 = self.metric(fea2)
        fea = torch.cat((fea1, fea2), 1)
        if sim:
            sim = self.cos(fea1, fea2)
            
            return sim
        else:
            return fea1,fea2
        
    def feature_sub(self, x1, x2):
        fea1 = self.net(x1)
        fea2 = self.net(x2)
        fea1 = self.metric(fea1)
        fea2 = self.metric(fea2)
        
        return fea1-fea2
        


class iresnet100_unsupervised:
    def __init__(self, conf, test=False):
        ################## for the design of the network: the embed_dim should divide the num_heads
        self.net =  iresnet100_cosface_(128)
        
        
        if len(conf.gpu.split(',')) > 1:
            self.net = nn.DataParallel(self.net)
        self.conf = conf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        if not test:
            self.init_train(conf)

    def init_train(self, conf):
        # torch.cuda.empty_cache()
        # self.opt = optim.AdamW(self.net.parameters(), lr=conf.lr)
        self.opt = optim.AdamW(filter(lambda p: p.requires_grad, self.net.parameters()), lr=conf.lr)
        self.ce = nn.CrossEntropyLoss()
    
    def nce_loss(self, imgs,batch_idx,temperature=0.07, mode='train'):
        # imgs, _ = batch
        # imgs = torch.cat(imgs, dim=0)

        # Encode all images
        feats = self.net(imgs)
        
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()
        ########### get ranking accuracy
        comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
                              cos_sim.masked_fill(pos_mask, -9e15)],
                             dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

        acc_top1 = (sim_argsort == 0).float().mean().item()
        acc_top5 = (sim_argsort < 5).float().mean().item()
        acc_mean_pos = 1+sim_argsort.float().mean().item()

        # print(batch_idx, '='*5,mode+'_acc_top1','%0.4f'%acc_top1 )
        # print(batch_idx, '='*5,mode+'_acc_top5','%0.4f'%acc_top5 )
        # print(batch_idx, '='*5,mode+'_acc_mean_pos', '%0.4f'%acc_mean_pos)

       
        return nll,acc_top1,acc_top5,acc_mean_pos

    def load(self, pth):
        checkpoints = torch.load(pth)
        self.net.load_state_dict(checkpoints['arch'])

    def save(self, pth):
        if len(self.conf.gpu.split(',')) > 1:
            torch.save({'arch': self.net.module.state_dict()}, '{}.pth'.format(pth))
        else:
            torch.save({'arch': self.net.state_dict()}, '{}.pth'.format(pth))

    def train(self, epoch, trainloader,writer):
        print("=====> Training epoch {}".format(epoch))
        t1_cout = AverageMeter()
        t5_cout = AverageMeter()
        m_cout = AverageMeter()
        self.net.train()
        for batch_idx, data in tqdm(enumerate(trainloader)):
            imgs,_ = data
            imgs = torch.cat(imgs, dim=0)
            imgs = imgs.to(self.device)
           
            self.opt.zero_grad()
            loss,acc_top1,acc_top5,acc_mean_pos = self.nce_loss(imgs,batch_idx,)
            loss.backward()
            self.opt.step()
            t1_cout.update(acc_top1, imgs.size(0))
            t5_cout.update(acc_top5, imgs.size(0))
            m_cout.update(acc_mean_pos, imgs.size(0))
            print('epoch: {} - iter:{} - loss:{}'.format(epoch, batch_idx, loss.item()))
        return loss,t1_cout.avg,t5_cout.avg,m_cout.avg
    


class metric_conv:
    def __init__(self, conf, test=False):
        self.net = iresnet100_metric(128)
        # self.net = attenNet2()
        if len(conf.gpu.split(',')) > 1:
            self.net = nn.DataParallel(self.net)
        self.conf = conf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.tp = None
        self.cs = None
        
        if not test:
            self.init_train(conf)
        self.total_cs_dis = []
        self.total_cs_lb = []
        self.feature_cs_ls =[]
        self.feature_cs_lb = []
    def init_train(self, conf):
        # torch.cuda.empty_cache()
        # self.opt = optim.AdamW(self.net.parameters(), lr=conf.lr)
        self.opt = optim.AdamW(filter(lambda p: p.requires_grad, self.net.parameters()), lr=conf.lr)
        self.mse = nn.MSELoss()

    def load(self, pth):
        checkpoints = torch.load(pth)
        self.net.load_state_dict(checkpoints['arch'])

    def save(self, pth):
        if len(self.conf.gpu.split(',')) > 1:
            torch.save({'arch': self.net.module.state_dict()}, '{}.pth'.format(pth))
        else:
            torch.save({'arch': self.net.state_dict()}, '{}.pth'.format(pth))

    def train(self, epoch, trainloader):
        ls_cout = AverageMeter()
        self.net.train()
        for batch_idx, data in tqdm(enumerate(trainloader)):
            x1, x2, label = data

            x1, x2, label = x1.to(self.device), x2.to(self.device), label.to(self.device)

            # x1 = x1.to(torch.float)
            # x2 = x2.to(torch.float)
            # print(x1)
            # print(x1.shape)
            self.opt.zero_grad()
            # print(label)
            out = self.net(x1, x2)
            
            loss = self.mse(out, label.to(torch.float32))
            loss.backward()
            self.opt.step()
            ls_cout.update(loss.item(), label.size(0))

        print("=====> Training epoch {}: average loss: {:.5f}".format(epoch, ls_cout.avg))
        return ls_cout.avg
    
    def gen_roc(self,fpr, tpr,cs_all=False):
        """Create a pyplot plot and save to buffer."""
        plt.figure()
        plt.xlabel("FPR", fontsize = 14)
        plt.ylabel("TPR", fontsize = 14)
        # plt.title("ROC Curve -{}-{}".format(self.tp,self.cs), fontsize = 14)
        plot = plt.plot(fpr, tpr, linewidth = 2)
        
        svpth = "./results/{}/roc/{}".format(self.conf.dataset,self.conf.model_name)
        if not os.path.exists(svpth):
            os.makedirs(svpth)
        if cs_all:
            plt.savefig(svpth+"/ROC-Curve-{}-cs-all".format(self.tp))
        else:
            plt.savefig(svpth+"/ROC-Curve-{}-{}".format(self.tp,self.cs))
        
        plt.close()
    
    
    def gen_hist(self,dist,lb,cs_all = False):
        """Create a pyplot plot and save to buffer."""
        plt.figure()
        plt.xlabel("Cosine similarity", fontsize = 14)
        plt.ylabel("Frequency", fontsize = 14)
        # plt.title("Histogram -{}-{}".format(self.tp,self.cs), fontsize = 14)
        plot = plt.hist(dist[lb==1], bins=30, range=(0,1), color='red', alpha=0.5, density=True, label='Pos')
        plot = plt.hist(dist[lb==0], bins=30, range=(0,1), color='blue', alpha=0.5, density=True, label='Neg')
        plt.legend(loc='upper right')
        
        svpth = "./results/{}/hist/{}".format(self.conf.dataset,self.conf.model_name)
        if not os.path.exists(svpth):
            os.makedirs(svpth)
        if cs_all:
            plt.savefig(svpth+"/Hist-{}-cs-all".format(self.tp))
        else:
            plt.savefig(svpth+"/Hist-{}-{}".format(self.tp,self.cs))
        
        plt.close()

    
    def test(self, testloader,vis_model =['roc']):
        self.net.eval()
        device = self.device
        total = 0
        correct = 0
        dis_ls = []
        lb_ls = []
        with torch.no_grad():
            for data in testloader:
                x1, x2, label = data
                x1, x2, label = x1.to(device), x2.to(device), label.to(device)
                # x1 = x1.to(torch.float)
                # x2 = x2.to(torch.float)
                out = self.net(x1, x2)
                pred = out>0.5
                pred = pred.to(torch.int)
                total += label.size(0)
                correct += (pred == label).sum().item()
                
                dis = out.cpu().numpy()
                dis_ls.append(dis)
                lb_ls.append(label.cpu().detach().numpy())
        
        dis_ls = np.concatenate(dis_ls,axis=0)
        lb_ls = np.concatenate(lb_ls,axis=0)     
        
        self.total_cs_dis.append(dis_ls)
        self.total_cs_lb.append(lb_ls)   
        
        if "roc" in vis_model:
            fpr, tpr, thresholds = roc_curve(lb_ls, dis_ls)
            self.gen_roc(fpr, tpr)
        if "hist" in vis_model:
            self.gen_hist(dis_ls,lb_ls)     
        acc = correct / total
        return acc
    
    
    def qualitative_result(self, testloader,vis_model =[]):
        self.net.eval()
        device = self.device
        total = 0
        correct = 0
        dis_ls = []
        lb_ls = []
        imgpth_ls = []
        pre_ls = []
        with torch.no_grad():
            for data in testloader:
                x1, x2, label,img1_pth,img2_pth = data
                x1, x2, label = x1.to(device), x2.to(device), label.to(device)
                # x1 = x1.to(torch.float)
                # x2 = x2.to(torch.float)
                out = self.net(x1, x2)
                pred = out>0.5
                pred = pred.to(torch.int)
                total += label.size(0)
                correct += (pred == label).sum().item()
                
                
                im1pth = img1_pth[0].split("/")[-1]
                im2pth = img2_pth[0].split("/")[-1]
                
                imgpth_ls.append([im1pth,im2pth])
                pre_ls.append(int(pred.cpu()))
                lb_ls.append(int(label.cpu().detach()))
        for i in range(len(imgpth_ls)):
            print(pre_ls[i],lb_ls[i],imgpth_ls[i])    
           
        acc = correct / total
        return acc
        
    def final_vis(self,vis_model=['roc','hist','tsne2d','tsne3d']):
        
        if "roc" in vis_model:
            fpr, tpr, thresholds = roc_curve(np.concatenate(self.total_cs_lb,axis=0), np.concatenate(self.total_cs_dis,axis=0))
            self.gen_roc(fpr, tpr,cs_all=True)
            pickle.dump([fpr, tpr, thresholds],open("./results/{}/roc/{}/{}-roc_all.pkl".format(self.conf.dataset,self.conf.model_name,self.tp),"wb"))
            
        if "hist" in vis_model:
            self.gen_hist(np.concatenate(self.total_cs_dis,axis=0),np.concatenate(self.total_cs_lb,axis=0),cs_all=True)
            
        if "tsne2d" in vis_model:
            self.tsne_vis(np.concatenate(self.feature_cs_ls,axis=0),np.concatenate(self.feature_cs_lb,axis=0),'2d',cs_all=True)
            
        if "tsne3d" in vis_model:
            self.tsne_vis(np.concatenate(self.feature_cs_ls,axis=0),np.concatenate(self.feature_cs_lb,axis=0),'3d',cs_all=True)
            
    def tsne_vis(self,fea_ls, lb_ls, vis_model='tsne2d',cs_all=False):
        if vis_model == '2d':
            Y = tsne(fea_ls, no_dims=2, initial_dims=64, perplexity=25.0, max_iter=1000)
            plt.figure()

            palette = sns.color_palette('pastel', 1)
            sns.scatterplot(x=Y[lb_ls==1][:,0],y=Y[lb_ls==1][:,1], hue=lb_ls[lb_ls==1], legend='full', palette=palette, alpha=0.5)

            palette = sns.color_palette('bright',4)
            sns.scatterplot(x=Y[lb_ls==0][:,0],y=Y[lb_ls==0][:,1], hue=lb_ls[lb_ls==0],legend='full',palette=palette,alpha=1)
            
            svpth = "./results/{}/tsne/{}".format(self.conf.dataset,self.conf.model_name)
            if not os.path.exists(svpth):
                os.makedirs(svpth)
            if cs_all:
                plt.savefig(svpth+"/tsne-{}-cs-all".format(self.tp))
            else:
                plt.savefig(svpth+"/tsne-{}-{}".format(self.tp,self.cs))
            
            plt.close()
        elif vis_model == '3d':
            Y = tsne(fea_ls, no_dims=3, initial_dims=64, perplexity=30.0, max_iter=1000)
            figure = plt.figure(figsize=(9,9))
            axes = figure.add_subplot(111,projection = "3d")
            dots = axes.scatter(xs = Y[:,0],ys = Y[:,1] ,zs =  Y[:,2],
                   c = lb_ls, cmap = plt.cm.get_cmap("Spectral",2))

            svpth = "./results/{}/tsne-3d/{}".format(self.conf.dataset,self.conf.model_name)
            if not os.path.exists(svpth):
                os.makedirs(svpth)
            if cs_all:
                plt.savefig(svpth+"/3dtsne-{}-cs-all".format(self.tp))
            else:
                plt.savefig(svpth+"/3dtsne-{}-{}".format(self.tp,self.cs))
            
            plt.close()
            
    def tsne(self, testloader,vis_model ='2d'):
        self.net.eval()
        device = self.device
        total = 0
        correct = 0
        fea_ls = []
        lb_ls = []
        with torch.no_grad():
            for data in testloader:
                x1, x2, label = data
                x1, x2, label = x1.to(device), x2.to(device), label.to(device)
                # x1 = x1.to(torch.float)
                # x2 = x2.to(torch.float)
                feature_sub = self.net.feature_sub(x1, x2)
                
                fea_ls.append(feature_sub.cpu().detach())
                lb_ls.append(label.cpu().detach().numpy())
        
        fea_ls = np.concatenate(fea_ls,axis=0)
        lb_ls = np.concatenate(lb_ls,axis=0)
        self.feature_cs_ls.append(fea_ls)
        self.feature_cs_lb.append(lb_ls)
        self.tsne_vis(fea_ls, lb_ls, vis_model) 
               
    def compare(self, trainloader,testloader):
        
        self.net.eval()
        device = self.device
        dis_ls = []
        lb_ls = []
        with torch.no_grad():
            for data in testloader:
                x1, x2, label = data
                x1, x2, label = x1.to(device), x2.to(device), label.to(device)
                fea1 = self.net.get_feature(x1)
                fea2 = self.net.get_feature(x2)
                dis = torch.nn.CosineSimilarity(dim=1)(fea1, fea2)
                dis = dis.cpu().numpy()
                dis_ls.append(dis)
                lb_ls.append(label.cpu().detach().numpy())
        
        dis_ls = np.concatenate(dis_ls,axis=0)
        lb_ls = np.concatenate(lb_ls,axis=0)
       
        
        pred = dis_ls > 0.5 ## set threshold
        acc = accuracy_score(lb_ls, pred)
        return acc,0.5
    
    def infer(self, data):
        self.net.eval()
        device = self.device
        with torch.no_grad():
            x1, x2, label = data
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)
            out = self.net(x1, x2)
            _, pred = torch.max(out.data, 1)
            return pred