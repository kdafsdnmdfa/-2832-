from module.videoswin import *
from utils.metric import *

class Video_unsupervised_Net_small:
    def __init__(self, conf, test=False,embed_dim= 48):
        ################## for the design of the network: the embed_dim should divide the num_heads
        if embed_dim ==96:
            self.net = SwinTransformer3D(embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads=[3, 6, 12, 24],
                                patch_size=(2, 4, 4),
                                window_size=(8, 7, 7),
                                drop_path_rate=0.1,
                                patch_norm=True)
        else:
            self.net = SwinTransformer3D(embed_dim=48, 
                                    depths=[2, 2, 6, 2],
                                    num_heads=[3, 6, 12, 24],
                                    patch_size=(2, 4, 4),
                                    window_size=(8, 7, 7),
                                    drop_path_rate=0.1,
                                    patch_norm=True)
        
        
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


        # mean or first x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        feats = feats.mean(dim=[2, 3, 4])
        # feats = feats[:,:,0,0,0]
        
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
            imgs = data
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

class Video_unsupervised_Net_small_fc:
    def __init__(self, conf, test=False,embed_dim=48):
        ################## for the design of the network: the embed_dim should divide the num_heads
        if embed_dim ==96:
            self.net = SwinTransformer3D(embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads=[3, 6, 12, 24],
                                patch_size=(2, 4, 4),
                                window_size=(8, 7, 7),
                                drop_path_rate=0.1,
                                patch_norm=True)
        else:
            self.net = SwinTransformer3D(embed_dim=48, 
                                    depths=[2, 2, 6, 2],
                                    num_heads=[3, 6, 12, 24],
                                    patch_size=(2, 4, 4),
                                    window_size=(8, 7, 7),
                                    drop_path_rate=0.1,
                                    patch_norm=True)
        
        self.fc = nn.Linear(384, 128)
        
        if len(conf.gpu.split(',')) > 1:
            self.net = nn.DataParallel(self.net)
        self.conf = conf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.fc.to(self.device)
        if not test:
            self.init_train(conf)

    def init_train(self, conf):
        # torch.cuda.empty_cache()
        # self.opt = optim.AdamW(self.net.parameters(), lr=conf.lr)
        # self.opt = optim.AdamW(filter(lambda p: p.requires_grad, self.net.parameters()), lr=conf.lr)
        self.opt = optim.AdamW([{'params':self.net.parameters()},
                                {'params':self.fc.parameters()}
                                ], lr=conf.lr)
        self.ce = nn.CrossEntropyLoss()
    
    def nce_loss(self, imgs,batch_idx,temperature=0.07, mode='train'):
        # imgs, _ = batch
        # imgs = torch.cat(imgs, dim=0)

        # Encode all images
        feats = self.net(imgs)


        # mean or first x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        feats = feats.mean(dim=[2, 3, 4])
        # feats = feats[:,:,0,0,0]
        feats = self.fc(feats)
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
            imgs = data
            imgs = torch.cat(imgs, dim=0)
            imgs = imgs.to(self.device)
           
            self.opt.zero_grad()
            loss,acc_top1,acc_top5,acc_mean_pos = self.nce_loss(imgs,batch_idx)
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