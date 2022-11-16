from module.videoswin import *
from utils.metric import *
# from utils.check_un_utils import evaluate_vit_distance
import numpy as np
from sklearn.metrics import accuracy_score,roc_curve
import sklearn
import matplotlib.pyplot as plt
import os
from utils.tsne import tsne
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import pickle

def load_model(net,model_path_name,model_path):
    if model_path_name in ['sup_ytb','un_nemo','un_ytb']:
        print('load pretrained model ', model_path)
        # checkpoint = torch.load('./results/sup_ytb/video_supervised_small_ytb/video_supervised_small_ytb_epoch45.pth')
        # checkpoint = torch.load('./results/un_nemo/video_un_small/video_un_small_epoch15.pth')
        # checkpoint = torch.load('/home/CODE/transformer/transformer-kin/results/un_ytb/video_un_small/video_un_small_epoch45.pth')
        checkpoint = torch.load(model_path)
        new_state_dict = OrderedDict()
        for k, v in checkpoint['arch'].items():
            name = k
            new_state_dict[name] = v

        net.load_state_dict(new_state_dict)
        return net
    
    elif model_path_name == '3DSWIN-ms1m':
        print('load pretrained model ', model_path)
        # checkpoint = torch.load('./results/3DSWIN-ms1m/Backbone_ms1m_vi_Epoch_21_Batch_102660_Time_2022-10-12-16-33_checkpoint.pth')
        checkpoint = torch.load(model_path)
        
        model_dict = net.state_dict()
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items(): 
            name = k[4:]
            if name in model_dict:
                new_state_dict[name] = v

        net.load_state_dict(new_state_dict)
        return net
    
    elif model_path_name == 'no_pretrain':
        print('no pretrained model')
        return net

class video_swin_pretrain_small(nn.Module):
    def __init__(self,model_path_name,model_path, embed_dim=48):
        super().__init__()
        if embed_dim == 48:
            self.swin_base = SwinTransformer3D(embed_dim=48,
                                        depths=[2, 2, 6, 2],
                                        num_heads=[3, 6, 12, 24],
                                        patch_size=(2, 4, 4),
                                        window_size=(8, 7, 7),
                                        drop_path_rate=0.1,
                                        patch_norm=True)
            self.f = nn.Linear(384, 64)
        else:   
            self.swin_base = SwinTransformer3D(embed_dim=96,
                            depths=[2, 2, 6, 2],
                            num_heads=[3, 6, 12, 24],
                            patch_size=(2, 4, 4),
                            window_size=(8, 7, 7),
                            drop_path_rate=0.1,
                            patch_norm=True)
            self.f = nn.Linear(768, 64)


        self.swin_base = load_model(self.swin_base,model_path_name,model_path)


        
        # self.f = nn.Linear(128,64)
        self.fc = nn.Sequential(
            nn.Linear(192, 64), #192
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.freeze(self.swin_base)
        
    def freeze(self, feat):
        for name, child in feat.named_children():
            # if name == 'repeat_3':
            #     return
            for param in child.parameters():
                param.requires_grad = False

    def forward(self,x1,x2):
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)
        fea1 = self.swin_base(x1)
        fea2 = self.swin_base(x2)
        # mean pooling
        fea1 = fea1.mean(dim=[2, 3, 4])  # [batch_size, hidden_dim]
        fea2 = fea2.mean(dim=[2, 3, 4])
        # fea1 = fea1[:,:,0,0,0]  
        # fea2 = fea2[:,:,0,0,0]
        fea1 = self.f(fea1)
        fea2 = self.f(fea2)
        fea_c1 = fea1**2-fea2**2
        fea_c2 = (fea1-fea2)**2
        fea_c3 = torch.mul(fea1,fea2)

        fea = torch.cat((fea_c1,fea_c2,fea_c3),1)
        pred = self.fc(fea)
        return pred
    

    
class VideoSwin_Net_pretrain:
    def __init__(self, conf, test=False):
        self.net = video_swin_pretrain_small(conf.model_path_name,conf.model_path, embed_dim=conf.embed_dim)
        # self.net = attenNet2()
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
            loss = self.ce(out, label)
            loss.backward()
            self.opt.step()
            ls_cout.update(loss.item(), label.size(0))

        print("=====> Training epoch {}: average loss: {:.5f}".format(epoch, ls_cout.avg))
        return ls_cout.avg

    def test(self, testloader):
        self.net.eval()
        device = self.device
        total = 0
        correct = 0
        with torch.no_grad():
            for data in testloader:
                x1, x2, label = data
                x1, x2, label = x1.to(device), x2.to(device), label.to(device)
                # x1 = x1.to(torch.float)
                # x2 = x2.to(torch.float)
                out = self.net(x1, x2)
                _, pred = torch.max(out.data, 1)
                total += label.size(0)
                correct += (pred == label).sum().item()
        acc = correct / total
        return acc

    def infer(self, data):
        self.net.eval()
        device = self.device
        with torch.no_grad():
            x1, x2, label = data
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)
            out = self.net(x1, x2)
            _, pred = torch.max(out.data, 1)
            return pred
        
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


################################################################################################ metirc
class video_swin_metric_pretrain(nn.Module):
    def __init__(self,model_path_name,model_path, embed_dim=48):
        super().__init__()

        if embed_dim == 48:
            self.swin_base = SwinTransformer3D(embed_dim=48,
                                        depths=[2, 2, 6, 2],
                                        num_heads=[3, 6, 12, 24],
                                        patch_size=(2, 4, 4),
                                        window_size=(8, 7, 7),
                                        drop_path_rate=0.1,
                                        patch_norm=True)
            self.metric = nn.Linear(384, 64)
        else:   
            self.swin_base = SwinTransformer3D(embed_dim=96,
                            depths=[2, 2, 6, 2],
                            num_heads=[3, 6, 12, 24],
                            patch_size=(2, 4, 4),
                            window_size=(8, 7, 7),
                            drop_path_rate=0.1,
                            patch_norm=True)
            self.metric = nn.Linear(768, 64)


        # print('load pretrained model ', './results/3DSWIN-ytb-pretrained-ms1m//Backbone_ms1m_vi_Epoch_125_Batch_9626_Time_2022-10-17-21-22_checkpoint.pth')
        # pretrained_dict = torch.load('./results/3DSWIN-ytb-pretrained-ms1m//Backbone_ms1m_vi_Epoch_125_Batch_9626_Time_2022-10-17-21-22_checkpoint.pth')
        # model_dict = self.swin_base.state_dict()
        # # 1. filter out unnecessary keys
        # pretrained_dict = {k[4:]: v for k, v in pretrained_dict.items() if k[4:] in model_dict}
        # # 2. overwrite entries in the existing state dict
        # model_dict.update(pretrained_dict)
        # self.swin_base.load_state_dict(pretrained_dict)
        self.swin_base = load_model(self.swin_base,model_path_name,model_path)


        # self.norm = nn.LayerNorm(64)
        # self.fc = nn.Linear(1, 2)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
        self.freeze(self.swin_base)

    def freeze(self, feat):
        for name, child in feat.named_children():
            # if name == 'repeat_3':
            #     return
            for param in child.parameters():
                param.requires_grad = False

    def forward(self,x1,x2,sim=True):
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)
        fea1 = self.swin_base(x1)
        fea2 = self.swin_base(x2)
        # mean pooling
        fea1 = fea1.mean(dim=[2, 3, 4])  # [batch_size, hidden_dim]
        fea2 = fea2.mean(dim=[2, 3, 4])
        
        fea1 = self.metric(fea1)
        fea2 = self.metric(fea2)
        # fea1 = F.normalize(fea1)
        # fea2 = F.normalize(fea2)
        if sim:
            sim = self.cos(fea1, fea2)
            
            return sim
        else:
            return fea1,fea2
    def get_feature(self,x):
        x = x.to(torch.float32)
        fea = self.swin_base(x)
        # mean pooling
        fea = fea.mean(dim=[2, 3, 4])
        return fea
    
    
    ### changed
    def feature_sub(self,x1,x2):
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)
        fea1 = self.swin_base(x1)
        fea2 = self.swin_base(x2)
        # mean pooling
        fea1 = fea1.mean(dim=[2, 3, 4])  # [batch_size, hidden_dim]
        fea2 = fea2.mean(dim=[2, 3, 4])
        
        fea1 = self.metric(fea1)
        fea2 = self.metric(fea2)
        return self.cos(fea1, fea2)
        
class metric_viswin:
    def __init__(self, conf, test=False):
        self.net = video_swin_metric_pretrain(conf.model_path_name,conf.model_path, embed_dim=conf.embed_dim)
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

    def init_train(self, conf):
        # torch.cuda.empty_cache()
        # self.opt = optim.AdamW(self.net.parameters(), lr=conf.lr)
        self.opt = optim.AdamW(filter(lambda p: p.requires_grad, self.net.parameters()), lr=conf.lr)
        self.ce = nn.MSELoss()

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
            
            loss = self.ce(out, label.to(torch.float32))
            loss.backward()
            self.opt.step()
            ls_cout.update(loss.item(), label.size(0))

        print("=====> Training epoch {}: average loss: {:.5f}".format(epoch, ls_cout.avg))
        return ls_cout.avg
    
    def gen_roc(self,fpr, tpr):
        """Create a pyplot plot and save to buffer."""
        plt.figure()
        plt.xlabel("FPR", fontsize = 14)
        plt.ylabel("TPR", fontsize = 14)
        plt.title("ROC Curve -{}-{}".format(self.tp,self.cs), fontsize = 14)
        plot = plt.plot(fpr, tpr, linewidth = 2)
        
        svpth = "./results/{}/roc/{}".format(self.conf.dataset,self.conf.model_name)
        if not os.path.exists(svpth):
            os.makedirs(svpth)
        plt.savefig(svpth+"/ROC Curve -{}-{}".format(self.tp,self.cs))
        
        plt.close()
    
    
    def gen_hist(self,dist,lb):
        """Create a pyplot plot and save to buffer."""
        plt.figure()
        plt.xlabel("Distance", fontsize = 14)
        plt.ylabel("Frequency", fontsize = 14)
        plt.title("Histogram -{}-{}".format(self.tp,self.cs), fontsize = 14)
        plot = plt.hist(dist[lb==1], bins=20, range=(0,1), color='red', alpha=0.5, density=True, label='Same')
        plot = plt.hist(dist[lb==0], bins=20, range=(0,1), color='blue', alpha=0.5, density=True, label='Diff')
        plt.legend(loc='upper right')
        
        svpth = "./results/{}/hist/{}".format(self.conf.dataset,self.conf.model_name)
        if not os.path.exists(svpth):
            os.makedirs(svpth)
        plt.savefig(svpth+"/Histogram -{}-{}".format(self.tp,self.cs))
        
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
        fpr, tpr, thresholds = roc_curve(lb_ls, dis_ls)
        if "roc" in vis_model:
            self.gen_roc(fpr, tpr)
        if "hist" in vis_model:
            self.gen_hist(dis_ls,lb_ls)     
        acc = correct / total
        return acc
    
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
            plt.savefig(svpth+"/tsne -{}-{}".format(self.tp,self.cs))
            
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
            plt.savefig(svpth+"/3dtsne-{}-{}".format(self.tp,self.cs))
            
            plt.close()
               
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

class metric_viswin:
    def __init__(self, conf, test=False):
        self.net = video_swin_metric_pretrain(conf.model_path_name,conf.model_path, embed_dim=conf.embed_dim)
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
        self.ce = nn.MSELoss()

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
            
            loss = self.ce(out, label.to(torch.float32))
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
        plot = plt.hist(dist[lb==0], bins=30, range=(0,1), color='darkorange', alpha=0.6, density=True, label='Neg')
        plot = plt.hist(dist[lb==1], bins=30, range=(0,1), color='forestgreen', alpha=0.6, density=True, label='Pos')
        
        plt.legend(loc='upper right')
        
        svpth = "./results/{}/hist/{}".format(self.conf.dataset,self.conf.model_name)
        if not os.path.exists(svpth):
            os.makedirs(svpth)
        if cs_all:
            plt.savefig(svpth+"/Hist-{}-cs-all".format(self.tp))
        else:
            plt.savefig(svpth+"/Hist-{}-{}".format(self.tp,self.cs))
        
        plt.close()

    
    def test(self, testloader,vis_model =[]):
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
            if len(fea_ls.shape)<2:
                fea_ls = np.expand_dims(fea_ls,axis=1)
            Y = tsne(fea_ls, no_dims=2, initial_dims=64, perplexity=30.0, max_iter=500)
            fig, ax = plt.subplots()

            # palette = sns.color_palette('pastel', 1)
            # sns.scatterplot(x=Y[lb_ls==1][:,0],y=Y[lb_ls==1][:,1], hue=lb_ls[lb_ls==1], legend='full', palette=palette, alpha=0.5)

            # palette = sns.color_palette('bright',4)
            # sns.scatterplot(x=Y[lb_ls==0][:,0],y=Y[lb_ls==0][:,1], hue=lb_ls[lb_ls==0],legend='full',palette=palette,alpha=1)
            
            ax.scatter(x=Y[lb_ls==0][:,0],y=Y[lb_ls==0][:,1], color = 'darkorange', label = 'Neg',alpha=0.3)
           
            ax.scatter(x=Y[lb_ls==1][:,0],y=Y[lb_ls==1][:,1], color = 'forestgreen', label = 'Pos',alpha=0.3)

            ax.legend(loc='upper right')
            
            
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


################ kinship identification ################

class video_swin_identification(nn.Module):
    def __init__(self,model_path_name,model_path, embed_dim=48):
        super().__init__()
        if embed_dim == 48:
            self.swin_base = SwinTransformer3D(embed_dim=48,
                                        depths=[2, 2, 6, 2],
                                        num_heads=[3, 6, 12, 24],
                                        patch_size=(2, 4, 4),
                                        window_size=(8, 7, 7),
                                        drop_path_rate=0.1,
                                        patch_norm=True)
            self.f = nn.Linear(384, 64)
        else:   
            self.swin_base = SwinTransformer3D(embed_dim=96,
                            depths=[2, 2, 6, 2],
                            num_heads=[3, 6, 12, 24],
                            patch_size=(2, 4, 4),
                            window_size=(8, 7, 7),
                            drop_path_rate=0.1,
                            patch_norm=True)
            self.f = nn.Linear(768, 64)


        self.swin_base = load_model(self.swin_base,model_path_name,model_path)


        
        # self.f = nn.Linear(128,64)
        self.fc = nn.Sequential(
            nn.Linear(192, 64), #192
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        self.freeze(self.swin_base)
        
    def freeze(self, feat):
        for name, child in feat.named_children():
            # if name == 'repeat_3':
            #     return
            for param in child.parameters():
                param.requires_grad = False

    def forward(self,x1,x2,getfeature=False):
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)
        fea1 = self.swin_base(x1)
        fea2 = self.swin_base(x2)
        # mean pooling
        fea1 = fea1.mean(dim=[2, 3, 4])  # [batch_size, hidden_dim]
        fea2 = fea2.mean(dim=[2, 3, 4])
        # fea1 = fea1[:,:,0,0,0]  
        # fea2 = fea2[:,:,0,0,0]
        fea1 = self.f(fea1)
        fea2 = self.f(fea2)
        fea_c1 = fea1**2-fea2**2
        fea_c2 = (fea1-fea2)**2
        fea_c3 = torch.mul(fea1,fea2)

        fea = torch.cat((fea_c1,fea_c2,fea_c3),1)
        if getfeature:
            return fea
        pred = self.fc(fea)
        return pred
    

    
class VideoSwin_Identification():
    def __init__(self, conf, test=False):
        print("VideoSwin_Identification")
        self.net = video_swin_identification(conf.model_path_name,conf.model_path, embed_dim=conf.embed_dim)
        # self.net = attenNet2()
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
            loss = self.ce(out, label)
            loss.backward()
            self.opt.step()
            ls_cout.update(loss.item(), label.size(0))

        print("=====> Training epoch {}: average loss: {:.5f}".format(epoch, ls_cout.avg))
        return ls_cout.avg

    def test(self, testloader):
        self.net.eval()
        device = self.device
        total = 0
        correct = 0
        with torch.no_grad():
            for data in testloader:
                x1, x2, label = data
                x1, x2, label = x1.to(device), x2.to(device), label.to(device)
                out = self.net(x1, x2)
                _, pred = torch.max(out.data, 1)
                total += label.size(0)
                correct += (pred == label).sum().item()
        acc = correct / total
        return acc

    def infer(self, data):
        self.net.eval()
        device = self.device
        with torch.no_grad():
            x1, x2, label = data
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)
            out = self.net(x1, x2)
            _, pred = torch.max(out.data, 1)
            return pred
        
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
                feature_sub = self.net(x1, x2,getfeature=True)
                
                fea_ls.append(feature_sub.cpu().detach())
                lb_ls.append(label.cpu().detach().numpy())
        
        fea_ls = np.concatenate(fea_ls,axis=0)
        lb_ls = np.concatenate(lb_ls,axis=0)        
        if vis_model == '2d':
            Y = tsne(fea_ls, no_dims=2, initial_dims=64, perplexity=25.0, max_iter=1000)
            plt.figure()

            palette = sns.color_palette('pastel', 5)
            sns.scatterplot(x=Y[:,0],y=Y[:,1], hue=lb_ls,legend='full',palette=palette,alpha=1)
            # sns.scatterplot(x=Y[lb_ls==1][:,0],y=Y[lb_ls==1][:,1], hue=lb_ls[lb_ls==1], legend='full', palette=palette, alpha=1)
            # sns.scatterplot(x=Y[lb_ls==2][:,0],y=Y[lb_ls==2][:,1], hue=lb_ls[lb_ls==2], legend='full', palette=palette, alpha=1)
            # sns.scatterplot(x=Y[lb_ls==3][:,0],y=Y[lb_ls==3][:,1], hue=lb_ls[lb_ls==3], legend='full', palette=palette, alpha=1)
            # sns.scatterplot(x=Y[lb_ls==4][:,0],y=Y[lb_ls==4][:,1], hue=lb_ls[lb_ls==4], legend='full', palette=palette, alpha=1)

            # palette = sns.color_palette('bright',4)
            
            svpth = "./results/{}/tsne/{}".format(self.conf.dataset,self.conf.model_name)
            if not os.path.exists(svpth):
                os.makedirs(svpth)
            plt.savefig(svpth+"/tsne -{}".format(self.cs))
            
            plt.close()
        elif vis_model == '3d':
            Y = tsne(fea_ls, no_dims=3, initial_dims=64, perplexity=30.0, max_iter=1000)
            figure = plt.figure(figsize=(9,9))
            axes = figure.add_subplot(111,projection = "3d")
            axes.scatter(xs = Y[lb_ls==0][:,0],ys = Y[lb_ls==0][:,1] ,zs =  Y[lb_ls==0][:,2],
                   c = 'r',label={'0': 'neg'})
            axes.scatter(xs = Y[lb_ls==1][:,0],ys = Y[lb_ls==1][:,1] ,zs =  Y[lb_ls==1][:,2],
                   c = 'b',label={'1': 'fd'})
            axes.scatter(xs = Y[lb_ls==2][:,0],ys = Y[lb_ls==2][:,1] ,zs =  Y[lb_ls==2][:,2],
                   c = 'g',label={'2': 'fs'})                        
            axes.scatter(xs = Y[lb_ls==3][:,0],ys = Y[lb_ls==3][:,1] ,zs =  Y[lb_ls==3][:,2],
                   c = 'orange',label={'3': 'md'})
            axes.scatter(xs = Y[lb_ls==4][:,0],ys = Y[lb_ls==4][:,1] ,zs =  Y[lb_ls==4][:,2],
                   c = 'purple',label={'4': 'ms'})
            axes.legend()            
            svpth = "./results/{}/tsne-3d/{}".format(self.conf.dataset,self.conf.model_name)
            if not os.path.exists(svpth):
                os.makedirs(svpth)
            plt.savefig(svpth+"/3dtsne-{}".format(self.cs))
            
            plt.close()     


class video_swin_identification2(nn.Module):
    def __init__(self,model_path_name,model_path, embed_dim=48):
        super().__init__()
        if embed_dim == 48:
            self.swin_base = SwinTransformer3D(embed_dim=48,
                                        depths=[2, 2, 6, 2],
                                        num_heads=[3, 6, 12, 24],
                                        patch_size=(2, 4, 4),
                                        window_size=(8, 7, 7),
                                        drop_path_rate=0.1,
                                        patch_norm=True)
            # self.f = nn.Linear(384, 64)
        else:   
            self.swin_base = SwinTransformer3D(embed_dim=96,
                            depths=[2, 2, 6, 2],
                            num_heads=[3, 6, 12, 24],
                            patch_size=(2, 4, 4),
                            window_size=(8, 7, 7),
                            drop_path_rate=0.1,
                            patch_norm=True)
            # self.f = nn.Linear(768, 64)


        self.swin_base = load_model(self.swin_base,model_path_name,model_path)


        
        # self.f = nn.Linear(128,64)
        self.fc = nn.Sequential(
            nn.Linear(768, 64), #192
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        self.freeze(self.swin_base)
        
    def freeze(self, feat):
        for name, child in feat.named_children():
            # if name == 'repeat_3':
            #     return
            for param in child.parameters():
                param.requires_grad = False

    def forward(self,x1,x2,getfeature=False):
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)
        fea1 = self.swin_base(x1)
        fea2 = self.swin_base(x2)
        # mean pooling
        fea1 = fea1.mean(dim=[2, 3, 4])  # [batch_size, hidden_dim]
        fea2 = fea2.mean(dim=[2, 3, 4])
        # fea1 = fea1[:,:,0,0,0]  
        # fea2 = fea2[:,:,0,0,0]
        # fea1 = self.f(fea1)
        # fea2 = self.f(fea2)
        # fea_c1 = fea1**2-fea2**2
        # fea_c2 = (fea1-fea2)**2
        # fea_c3 = torch.mul(fea1,fea2)

        fea = torch.cat((fea1,fea2),1)
        if getfeature:
            return fea
        pred = self.fc(fea)
        return pred
    
class VideoSwin_Identification2():
    def __init__(self, conf, test=False):
        self.net = video_swin_identification2(conf.model_path_name,conf.model_path, embed_dim=conf.embed_dim)
        # self.net = attenNet2()
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
            loss = self.ce(out, label)
            loss.backward()
            self.opt.step()
            ls_cout.update(loss.item(), label.size(0))

        print("=====> Training epoch {}: average loss: {:.5f}".format(epoch, ls_cout.avg))
        return ls_cout.avg

    def test(self, testloader):
        self.net.eval()
        device = self.device
        total = 0
        correct = 0
        with torch.no_grad():
            for data in testloader:
                x1, x2, label = data
                x1, x2, label = x1.to(device), x2.to(device), label.to(device)
                out = self.net(x1, x2)
                _, pred = torch.max(out.data, 1)
                total += label.size(0)
                correct += (pred == label).sum().item()
        acc = correct / total
        return acc

    def infer(self, data):
        self.net.eval()
        device = self.device
        with torch.no_grad():
            x1, x2, label = data
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)
            out = self.net(x1, x2)
            _, pred = torch.max(out.data, 1)
            return pred
        
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
                feature_sub = self.net(x1, x2,getfeature=False)
                
                fea_ls.append(feature_sub.cpu().detach())
                lb_ls.append(label.cpu().detach().numpy())
        
        fea_ls = np.concatenate(fea_ls,axis=0)
        lb_ls = np.concatenate(lb_ls,axis=0)        
        if vis_model == '2d':
            Y = tsne(fea_ls, no_dims=2, initial_dims=64, perplexity=25.0, max_iter=1000)
            plt.figure()

            palette = sns.color_palette('pastel', 5)
            sns.scatterplot(x=Y[:,0],y=Y[:,1], hue=lb_ls,legend='full',palette=palette,alpha=1)
            # sns.scatterplot(x=Y[lb_ls==1][:,0],y=Y[lb_ls==1][:,1], hue=lb_ls[lb_ls==1], legend='full', palette=palette, alpha=1)
            # sns.scatterplot(x=Y[lb_ls==2][:,0],y=Y[lb_ls==2][:,1], hue=lb_ls[lb_ls==2], legend='full', palette=palette, alpha=1)
            # sns.scatterplot(x=Y[lb_ls==3][:,0],y=Y[lb_ls==3][:,1], hue=lb_ls[lb_ls==3], legend='full', palette=palette, alpha=1)
            # sns.scatterplot(x=Y[lb_ls==4][:,0],y=Y[lb_ls==4][:,1], hue=lb_ls[lb_ls==4], legend='full', palette=palette, alpha=1)

            # palette = sns.color_palette('bright',4)
            
            svpth = "./results/{}/tsne/{}".format(self.conf.dataset,self.conf.model_name)
            if not os.path.exists(svpth):
                os.makedirs(svpth)
            plt.savefig(svpth+"/tsne -{}".format(self.cs))
            
            plt.close()
        elif vis_model == '3d':
            Y = tsne(fea_ls, no_dims=3, initial_dims=5, perplexity=30.0, max_iter=500)
            figure = plt.figure(figsize=(9,9))
            axes = figure.add_subplot(111,projection = "3d")
            # axes.scatter(xs = Y[lb_ls==0][:,0],ys = Y[lb_ls==0][:,1] ,zs =  Y[lb_ls==0][:,2],
            #        c = 'r',label={'0': 'neg'})
            axes.scatter(xs = Y[lb_ls==1][:,0],ys = Y[lb_ls==1][:,1] ,zs =  Y[lb_ls==1][:,2],
                   c = 'b',label={'1': 'fd'})
            axes.scatter(xs = Y[lb_ls==2][:,0],ys = Y[lb_ls==2][:,1] ,zs =  Y[lb_ls==2][:,2],
                   c = 'g',label={'2': 'fs'})                        
            axes.scatter(xs = Y[lb_ls==3][:,0],ys = Y[lb_ls==3][:,1] ,zs =  Y[lb_ls==3][:,2],
                   c = 'orange',label={'3': 'md'})
            axes.scatter(xs = Y[lb_ls==4][:,0],ys = Y[lb_ls==4][:,1] ,zs =  Y[lb_ls==4][:,2],
                   c = 'purple',label={'4': 'ms'})
            axes.legend()            
            svpth = "./results/{}/tsne-3d/{}".format(self.conf.dataset,self.conf.model_name)
            if not os.path.exists(svpth):
                os.makedirs(svpth)
            plt.savefig(svpth+"/3dtsne-{}".format(self.cs))
            
            plt.close()   


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
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward_hk(nn.Module):
    def __init__(self, dim, hidden_dim,out_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim*2 , hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention_hk(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x,k):

        qv = self.to_qkv(x).chunk(2, dim = -1)
        q,v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qv)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class PreNorm_hk(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x,k):
        return self.fn(self.norm(x),k )
class KAT(nn.Module):
    """
     kinship-aware-transformer (KAT)
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim*2, FeedForward_hk(dim, mlp_dim,heads*dim_head, dropout=dropout)),
                PreNorm_hk(dim, Attention_hk(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x1,x2):
        # for get_k, attn, ff in self.layers:
        #     k = get_k(x)
        #     x = attn(x,k) + x
        #     x = ff(x) + x
        for get_k, attn, ff in self.layers:
            x_k = torch.cat((x1,x2),2)
            k = get_k(x_k)
            ## attentioned x1
            x1 = attn(x1,k) + x1
            x1 = ff(x1) + x1
            ## attentioned x2
            x2 = attn(x2,k)+x2
            x2 = ff(x2)+x2

        return x1,x2
class Transformer_enhance(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    
    
##################################  
class video_swin_identification3(nn.Module):
    def __init__(self,model_path_name,model_path, embed_dim=48):
        super().__init__()
        if embed_dim == 48:
            self.swin_base = SwinTransformer3D(embed_dim=48,
                                        depths=[2, 2, 6, 2],
                                        num_heads=[3, 6, 12, 24],
                                        patch_size=(2, 4, 4),
                                        window_size=(8, 7, 7),
                                        drop_path_rate=0.1,
                                        patch_norm=True)
            # self.f = nn.Linear(384, 64)
        else:   
            self.swin_base = SwinTransformer3D(embed_dim=96,
                            depths=[2, 2, 6, 2],
                            num_heads=[3, 6, 12, 24],
                            patch_size=(2, 4, 4),
                            window_size=(8, 7, 7),
                            drop_path_rate=0.1,
                            patch_norm=True)
            # self.f = nn.Linear(768, 64)


        self.swin_base = load_model(self.swin_base,model_path_name,model_path)


        self.trans_atten = Transformer_enhance(dim=768, depth=2, heads=8, dim_head=64, mlp_dim=256, dropout=0.1)
        # self.f = nn.Linear(128,64)
        self.fc = nn.Sequential(
            nn.Linear(768, 64), #192
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        self.freeze(self.swin_base)
        
    def freeze(self, feat):
        for name, child in feat.named_children():
            # if name == 'repeat_3':
            #     return
            for param in child.parameters():
                param.requires_grad = False

    def forward(self,x1,x2,getfeature=False):
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)
        fea1 = self.swin_base(x1)
        fea2 = self.swin_base(x2)
        # mean pooling
        fea1 = fea1.mean(dim=[2, 3, 4])  # [batch_size, hidden_dim]
        fea2 = fea2.mean(dim=[2, 3, 4])
        # fea1 = fea1[:,:,0,0,0]  
        # fea2 = fea2[:,:,0,0,0]
        # fea1 = self.f(fea1)
        # fea2 = self.f(fea2)
        # fea_c1 = fea1**2-fea2**2
        # fea_c2 = (fea1-fea2)**2
        # fea_c3 = torch.mul(fea1,fea2)

        fea = torch.cat((fea1,fea2),1)
        fea = torch.unsqueeze(fea, 1)
        fea = self.trans_atten(fea).squeeze()
        if getfeature:
            return fea
        pred = self.fc(fea)
        return pred

class VideoSwin_Identification3(VideoSwin_Identification):
    def __init__(self, conf, test=False):
        print('VideoSwin_Identification3')
        self.net = video_swin_identification3(conf.model_path_name,conf.model_path, embed_dim=conf.embed_dim)
        # self.net = attenNet2()
        if len(conf.gpu.split(',')) > 1:
            self.net = nn.DataParallel(self.net)
        self.conf = conf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        if not test:
            self.init_train(conf)
                             
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
    img1 = torch.randn((1,3, 144, 224, 224))
    img2 = torch.randn((1, 3, 144, 224, 224))
    # import torchvision.transforms.functional as F
    #
    # tt = torch.randn((3, 144, 1024, 960))
    #
    # tt_resize = F.resize(tt,[224,224])
    # tt_resize = tt_resize.unsqueeze(0)
    # net = video_swin()
    # pred = net(img1, img2)

    # print(pred.size())