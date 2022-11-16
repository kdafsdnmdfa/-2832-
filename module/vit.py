from cmath import cos
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from torch.nn import Parameter
# from IPython import embed
import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.metric import AverageMeter
import matplotlib.pyplot as plt
import os
from utils.tsne import tsne
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict
import pickle

MIN_NUM_PATCHES = 16

class Softmax(nn.Module):
    r"""Implement of Softmax (normal classification head):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
        """
    def __init__(self, in_features, out_features, device_id):
        super(Softmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input, label):
        if self.device_id == None:
            x = input
            out = F.linear(x, self.weight, self.bias)
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            sub_biases = torch.chunk(self.bias, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            bias = sub_biases[0].cuda(self.device_id[0])
            out = F.linear(temp_x, weight, bias)
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                bias = sub_biases[i].cuda(self.device_id[i])
                out = torch.cat((out, F.linear(temp_x, weight, bias).cuda(self.device_id[0])), dim=1)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()



class ArcFace(nn.Module):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
            s: norm of input feature
            m: margin
            cos(theta+m)
        """

    def __init__(self, in_features, out_features, device_id, s=64.0, m=0.50, easy_margin=False):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id

        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])),
                                   dim=1)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output


class CosFace(nn.Module):
    r"""Implement of CosFace (https://arxiv.org/pdf/1801.09414.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
        s: norm of input feature
        m: margin
        cos(theta)-m
    """

    def __init__(self, in_features, out_features, device_id, s=64.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id
        self.s = s
        self.m = m
        print("self.device_id", self.device_id)
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------

        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])),
                                   dim=1)
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot

        one_hot.scatter_(1, label.cuda(self.device_id[0]).view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', s = ' + str(self.s) \
               + ', m = ' + str(self.m) + ')'

class SFaceLoss(nn.Module):

    def __init__(self, in_features, out_features, device_id, s = 64.0, k = 80.0, a = 0.80, b = 1.22):
        super(SFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id
        self.s = s
        self.k = k
        self.a = a
        self.b = b
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        #nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.weight, gain=2, mode='out')

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))

            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])), dim=1)
        # --------------------------- s*cos(theta) ---------------------------
        output = cosine * self.s
        # --------------------------- sface loss ---------------------------

        one_hot = torch.zeros(cosine.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        one_hot.scatter_(1, label.view(-1, 1), 1)

        zero_hot = torch.ones(cosine.size())
        if self.device_id != None:
            zero_hot = zero_hot.cuda(self.device_id[0])
        zero_hot.scatter_(1, label.view(-1, 1), 0)

        WyiX = torch.sum(one_hot * output, 1)
        with torch.no_grad():
            # theta_yi = torch.acos(WyiX)
            theta_yi = torch.acos(WyiX / self.s)
            weight_yi = 1.0 / (1.0 + torch.exp(-self.k * (theta_yi - self.a)))
        intra_loss = - weight_yi * WyiX

        Wj = zero_hot * output
        with torch.no_grad():
            # theta_j = torch.acos(Wj)
            theta_j = torch.acos(Wj / self.s)
            weight_j = 1.0 / (1.0 + torch.exp(self.k * (theta_j - self.b)))
        inter_loss = torch.sum(weight_j * Wj, 1)

        loss = intra_loss.mean() + inter_loss.mean()
        Wyi_s = WyiX / self.s
        Wj_s = Wj / self.s
        return output, loss, intra_loss.mean(), inter_loss.mean(), Wyi_s.mean(), Wj_s.mean()



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        #embed()
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)

        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            #embed()
            x = ff(x)
        return x


class ViT_face(nn.Module):
    def __init__(self, *, loss_type, GPU_ID, num_class, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
        )
        self.loss_type = loss_type
        self.GPU_ID = GPU_ID
        if self.loss_type == 'None':
            print("no loss for vit_face")
        else:
            if self.loss_type == 'Softmax':
                self.loss = Softmax(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
            elif self.loss_type == 'CosFace':
                self.loss = CosFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
            elif self.loss_type == 'ArcFace':
                self.loss = ArcFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
            elif self.loss_type == 'SFace':
                self.loss = SFaceLoss(in_features=dim, out_features=num_class, device_id=self.GPU_ID)


    def forward(self, img, label= None , mask = None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        # print(x)
        # print(x.shape)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, mask)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        emb = self.mlp_head(x)
        if label is not None:
            x = self.loss(emb, label)
            return x, emb
        else:
            return emb




class res_unit(nn.Module):
    """
    this is the attention module before Residual structure
    """
    def __init__(self,channel,up_size = None):
        """

        :param channel: channels of input feature map
        :param up_size: upsample size
        """
        super(res_unit,self).__init__()
        self.pool = nn.MaxPool2d(2,2)
        self.conv = nn.Conv2d(channel,channel,3,padding=1)
        if up_size == None:
            self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        else:
            self.upsample = nn.Upsample(size=(up_size,up_size), mode='bilinear', align_corners=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        identity = x
        x = self.pool(x)
        x = self.conv(x)
        x = self.upsample(x)
        x = self.sigmoid(x)
        x = torch.mul(identity,x)
        return x




# from utils.check_un_utils import evaluate_vit_distance
import numpy as np
from sklearn.metrics import accuracy_score,roc_curve
import sklearn





class vit_unsupervised:
    def __init__(self, conf, test=False):
        ################## for the design of the network: the embed_dim should divide the num_heads
        self.net = ViT_face(
            loss_type=None,
            GPU_ID=[0],
            num_class=0,
            image_size=64,  # 112
            patch_size=8,
            dim=128,  # 512
            depth=20,  # 20
            heads=8,
            mlp_dim=512,  # 2048
            dropout=0.1,
            emb_dropout=0.1)
        
        
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
        
    def test(self, testloader):
        pass
       
    def infer(self, data):
        pass


class vit_metric_pretrain(nn.Module):
    def __init__(self):
        super(vit_metric_pretrain, self).__init__()
        self.vit_base = ViT_face(
            loss_type=None,
            GPU_ID=[0],
            num_class=0,
            image_size=64,  # 112
            patch_size=8,
            dim=128,  # 512
            depth=20,  # 20
            heads=8,
            mlp_dim=512,  # 2048
            dropout=0.1,
            emb_dropout=0.1)
        # pretrained_dict = torch.load('./Backbone_VIT_Epoch_2_Batch_20000_Time_2021-01-12-16-48_checkpoint.pth') # 112
        # pretrained_dict = torch.load('./Backbone_VIT_Epoch_15_Batch_156080_Time_2022-07-30-11-05_checkpoint.pth')  # 64
        # pretrained_dict = torch.load('./results/un_ms1m/vit_un/vit_un_epoch0.pth')  # 64
        
        # model_dict = self.vit_base.state_dict()
        # # 1. filter out unnecessary keys
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # # 2. overwrite entries in the existing state dict
        # model_dict.update(pretrained_dict)
        # self.vit_base.load_state_dict(pretrained_dict)
        
        checkpoint = torch.load('./results/un_ms1m/vit_un/vit_un_epoch0.pth')
        new_state_dict = OrderedDict()
        for k, v in checkpoint['arch'].items():
            name = k
            new_state_dict[name] = v

        self.vit_base.load_state_dict(new_state_dict)
        
        
        self.freeze(self.vit_base)
        self.metric = nn.Linear(128, 64)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)


    def freeze(self, feat):
        for name, child in feat.named_children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x1, x2,sim = True):
        fea1 = self.vit_base(x1)
        fea2 = self.vit_base(x2)
        fea1 = self.metric(fea1)
        fea2 = self.metric(fea2)
        fea = torch.cat((fea1, fea2), 1)
        if sim:
            sim = self.cos(fea1, fea2)
            
            return sim
        else:
            return fea1,fea2
    def feature_sub(self, x1, x2):
        fea1 = self.vit_base(x1)
        fea2 = self.vit_base(x2)
        fea1 = self.metric(fea1)
        fea2 = self.metric(fea2)
        
        return (fea1-fea2)**2
        


class metric_vit:
    def __init__(self, conf, test=False):
        self.net = vit_metric_pretrain()
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
if __name__ =='__main__':
    # net = ViT_face(
    #     loss_type='None',
    #     GPU_ID=[0],
    #     num_class=2,
    #     image_size=112,
    #     patch_size=8,
    #     dim=512,
    #     depth=20,
    #     heads=8,
    #     mlp_dim=2048,
    #     dropout=0.1,
    #     emb_dropout=0.1
    # )
    # img = torch.randn(1, 3, 112, 112)
    #
    # preds = net(img)  # (1, 1000)
    # print(preds)
    pass