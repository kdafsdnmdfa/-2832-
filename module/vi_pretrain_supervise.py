from module.videoswin import *
from utils.metric import *

  
class Video_supervise_Net:
    def __init__(self, conf, test=False):
        ################## for the design of the network: the embed_dim should divide the num_heads
        self.net = SwinTransformer3D(embed_dim=96, 
                                  depths=[2, 2, 6, 2],
                                  num_heads=[3, 6, 12, 24],
                                  patch_size=(2, 4, 4),
                                  window_size=(8, 7, 7),
                                  drop_path_rate=0.1,
                                  patch_norm=True)
        
         # checkpoint = torch.load('./checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth')
        # # print('load pretrained model ', './results/ytf/video_un/video_un_epoch5.pth')
        # # checkpoint = torch.load('./results/ytf/video_un/video_un_epoch5.pth')
        # new_state_dict = OrderedDict()
        # for k, v in checkpoint['state_dict'].items():
        #     if 'backbone' in k:
        #         name = k[9:]
        #         new_state_dict[name] = v

        # self.swin_base.load_state_dict(new_state_dict)
        
        
        if len(conf.gpu.split(',')) > 1:
            self.net = nn.DataParallel(self.net)
        self.conf = conf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        if not test:
            self.init_train(conf)
        self.fc = nn.Sequential(
            nn.Linear(1536, 64), 
            nn.ReLU(),
            nn.Linear(64, 12)
        )
        self.fc.to(self.device)
    def init_train(self, conf):
       
        # self.opt = optim.AdamW(filter(lambda p: p.requires_grad, self.net.parameters()), lr=conf.lr)
        self.opt = optim.AdamW([{'params':filter(lambda p: p.requires_grad, self.net.parameters())},
                                {'params':self.fc.parameters()}
                                ], lr=conf.lr)
        
        
        self.ce = nn.CrossEntropyLoss()
        
    
    def identification(self, x1, x2, mode='train'):
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)
        fea1 = self.net(x1)
        fea2 = self.net(x2)
        fea1 = fea1.mean(dim=[2, 3, 4])  # [batch_size, hidden_dim]
        fea2 = fea2.mean(dim=[2, 3, 4])
        fea = torch.cat([fea1, fea2], dim=1)
        pred = self.fc(fea)
        return pred
        
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

            self.opt.zero_grad()
            # print(label)
            out = self.identification(x1, x2)
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
                out = self.identification(x1, x2)
                _, pred = torch.max(out.data, 1)
                total += label.size(0)
                correct += (pred == label).sum().item()
        acc = correct / total
        return acc
       
    def infer(self, data):
        pass

class Video_supervise_Net_small(Video_supervise_Net):
    def __init__(self, conf, test=False):
        ################## for the design of the network: the embed_dim should divide the num_heads
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
        self.fc = nn.Sequential(
            nn.Linear(768, 64), #192
            nn.ReLU(),
            nn.Linear(64, 12)
        )
        self.fc.to(self.device)
        if not test:
            self.init_train(conf)
        
        
class Video_supervise_Net_small_ytb:
    def __init__(self, conf, test=False):
        ################## for the design of the network: the embed_dim should divide the num_heads
        self.net = SwinTransformer3D(embed_dim=48, 
                                  depths=[2, 2, 6, 2],
                                  num_heads=[3, 6, 12, 24],
                                  patch_size=(2, 4, 4),
                                  window_size=(8, 7, 7),
                                  drop_path_rate=0.1,
                                  patch_norm=True)
        # self.net = SwinTransformer3D(embed_dim=96, 
        #                           depths=[2, 2, 6, 2],
        #                           num_heads=[3, 6, 12, 24],
        #                           patch_size=(2, 4, 4),
        #                           window_size=(8, 7, 7),
        #                           drop_path_rate=0.1,
        #                           patch_norm=True)
        # checkpoint = torch.load('./checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth')
        # # print('load pretrained model ', './results/ytf/video_un/video_un_epoch5.pth')
        # # checkpoint = torch.load('./results/ytf/video_un/video_un_epoch5.pth')
        # new_state_dict = OrderedDict()
        # for k, v in checkpoint['state_dict'].items():
        #     if 'backbone' in k:
        #         name = k[9:]
        #         new_state_dict[name] = v

        # self.net.load_state_dict(new_state_dict)
        
    
        # if len(conf.gpu.split(',')) > 1:
        #     self.net = nn.DataParallel(self.net)
            
            
            
        self.conf = conf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.fc = nn.Sequential(
            nn.Linear(384, 64), 
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.fc.to(self.device)
        if not test:
            self.init_train(conf)
        
        
    def init_train(self, conf):
       
        # self.opt = optim.AdamW(filter(lambda p: p.requires_grad, self.net.parameters()), lr=conf.lr)
        self.opt = optim.AdamW([{'params':filter(lambda p: p.requires_grad, self.net.parameters())},
                                {'params':self.fc.parameters()}
                                ], lr=conf.lr)
        
        
        self.ce = nn.CrossEntropyLoss()
        
    
    def identification(self, x1, x2, mode='train'):
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)
        fea1 = self.net(x1)
        fea2 = self.net(x2)
        fea1 = fea1.mean(dim=[2, 3, 4])  # [batch_size, hidden_dim]
        fea2 = fea2.mean(dim=[2, 3, 4])
        # fea = torch.cat([fea1, fea2], dim=1)
        fea = torch.absolute(fea1 - fea2)
        pred = self.fc(fea)
        return pred
    
    def cal_similarity(self, x1,x2):
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)
        fea1 = self.net(x1)
        fea2 = self.net(x2)
        fea1 = fea1.mean(dim=[2, 3, 4])  # [batch_size, hidden_dim]
        fea2 = fea2.mean(dim=[2, 3, 4])
        distance = nn.CosineSimilarity(dim=1, eps=1e-6)
        fea1 = F.normalize(fea1, p=2, dim=1)
        fea2 = F.normalize(fea2, p=2, dim=1)
        cos_dist = distance(fea1, fea2)
        # cos_dist = torch.sum(torch.absolute(fea1 - fea2)**2,dim=1)
        return cos_dist 
    
    def cal_accuracy(self,cos_dist,label):
        best_acc = 0
        best_th = 0
        for i in range(len(cos_dist)):
            th = cos_dist[i]
            y_test = cos_dist>=th
            acc = np.mean((y_test==label).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_th = th
        for  i in range(len(cos_dist)):
            if cos_dist[i] >= best_th:
                cos_dist[i] = 1
            else:
                cos_dist[i] = 0
        normal_acc = np.mean((cos_dist==label).astype(int))
            
        return best_acc, best_th,normal_acc
    
    def test_distance(self, testloader):
        self.net.eval()
        device = self.device
        distance_ls = torch.tensor([])
        label_ls = torch.tensor([])
        with torch.no_grad():
            for data in testloader:
                x1, x2, label = data
                x1, x2, label = x1.to(device), x2.to(device), label
                cos_dist = self.cal_similarity(x1, x2).cpu()
                distance_ls = torch.cat([distance_ls, cos_dist], dim=0)
                label_ls = torch.cat([label_ls, label], dim=0)
        
        distance_ls = distance_ls.numpy()
        label_ls = label_ls.numpy()
        best_acc, best_th,normal_acc = self.cal_accuracy(distance_ls, label_ls)
        return best_acc, best_th,normal_acc
    
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

            self.opt.zero_grad()
            # print(label)
            out = self.identification(x1, x2)
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
        distance_ls = []
        with torch.no_grad():
            for data in testloader:
                x1, x2, label = data
                x1, x2, label = x1.to(device), x2.to(device), label.to(device)
                # x1 = x1.to(torch.float)
                # x2 = x2.to(torch.float)
                out = self.identification(x1, x2)
                _, pred = torch.max(out.data, 1)
                total += label.size(0)
                correct += (pred == label).sum().item()
        acc = correct / total
        return acc
       
    def infer(self, data):
        pass   





from .vit import CosFace
from sklearn.metrics import accuracy_score,roc_curve
class Video_supervise_Net_small_ytb_classification:
    def __init__(self, conf, test=False):
        ################## for the design of the network: the embed_dim should divide the num_heads
        self.net = SwinTransformer3D(embed_dim=48, 
                                  depths=[2, 2, 6, 2],
                                  num_heads=[3, 6, 12, 24],
                                  patch_size=(2, 4, 4),
                                  window_size=(8, 7, 7),
                                  drop_path_rate=0.1,
                                  patch_norm=True)

        self.fc = CosFace(in_features=384, out_features=1276, device_id=None)
        self.conf = conf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        
        if not test:
            self.init_train(conf)
        
        
    def init_train(self, conf):
       
        # self.opt = optim.AdamW(filter(lambda p: p.requires_grad, self.net.parameters()), lr=conf.lr)
        self.opt = optim.AdamW([{'params':filter(lambda p: p.requires_grad, self.net.parameters())},
                                {'params':self.fc.parameters()}
                                ], lr=conf.lr)
        
        
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
            x,  label = data

            x,  label = x.to(self.device), label.to(self.device)

            self.opt.zero_grad()
            # print(label)
            x = self.net(x)
            out,_ = self.fc(x, label)
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
        distance_ls = []
        with torch.no_grad():
            for data in testloader:
                x1, x2, label = data
                x1, x2, label = x1.to(device), x2.to(device), label.to(device)
                # x1 = x1.to(torch.float)
                # x2 = x2.to(torch.float)
                out = self.identification(x1, x2)
                _, pred = torch.max(out.data, 1)
                total += label.size(0)
                correct += (pred == label).sum().item()
        acc = correct / total
        return acc
    
    def compare(self, testloader):
        
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
        pass   
    
    
class Video_supervise_Net_kinect_ytb:
    def __init__(self, conf, test=False):
        ################## for the design of the network: the embed_dim should divide the num_heads
        self.net = SwinTransformer3D(embed_dim=96, 
                                  depths=[2, 2, 6, 2],
                                  num_heads=[3, 6, 12, 24],
                                  patch_size=(2, 4, 4),
                                  window_size=(8, 7, 7),
                                  drop_path_rate=0.1,
                                  patch_norm=True)
        checkpoint = torch.load('./checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth')
        # print('load pretrained model ', './results/ytf/video_un/video_un_epoch5.pth')
        # checkpoint = torch.load('./results/ytf/video_un/video_un_epoch5.pth')
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if 'backbone' in k:
                name = k[9:]
                new_state_dict[name] = v

        self.net.load_state_dict(new_state_dict)
        
        if len(conf.gpu.split(',')) > 1:
            self.net = nn.DataParallel(self.net)
        self.conf = conf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.fc = nn.Sequential(
            nn.Linear(384, 64), 
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.fc.to(self.device)
        if not test:
            self.init_train(conf)
        
        
    def init_train(self, conf):
       
        # self.opt = optim.AdamW(filter(lambda p: p.requires_grad, self.net.parameters()), lr=conf.lr)
        self.opt = optim.AdamW([{'params':filter(lambda p: p.requires_grad, self.net.parameters())},
                                {'params':self.fc.parameters()}
                                ], lr=conf.lr)
        
        
        self.ce = nn.CrossEntropyLoss()
        
    
    def identification(self, x1, x2, mode='train'):
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)
        fea1 = self.net(x1)
        fea2 = self.net(x2)
        fea1 = fea1.mean(dim=[2, 3, 4])  # [batch_size, hidden_dim]
        fea2 = fea2.mean(dim=[2, 3, 4])
        # fea = torch.cat([fea1, fea2], dim=1)
        fea = torch.absolute(fea1 - fea2)
        pred = self.fc(fea)
        return pred
    
    def cal_similarity(self, x1,x2):
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)
        fea1 = self.net(x1)
        fea2 = self.net(x2)
        fea1 = fea1.mean(dim=[2, 3, 4])  # [batch_size, hidden_dim]
        fea2 = fea2.mean(dim=[2, 3, 4])
        # fea1 = fea1[:,:,0,0,0]  
        # fea2 = fea2[:,:,0,0,0]
        distance = nn.CosineSimilarity(dim=1, eps=1e-6)
        # fea1 = F.normalize(fea1, p=2, dim=1)
        # fea2 = F.normalize(fea2, p=2, dim=1)
        cos_dist = distance(fea1, fea2)
        # cos_dist = torch.sum(torch.absolute(fea1 - fea2)**2,dim=1)
        return cos_dist 
    
    def cal_accuracy(self,cos_dist,label):
        best_acc = 0
        best_th = 0
        for i in range(len(cos_dist)):
            th = cos_dist[i]
            y_test = cos_dist>=th
            acc = np.mean((y_test==label).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_th = th
        for  i in range(len(cos_dist)):
            if cos_dist[i] >= best_th:
                cos_dist[i] = 1
            else:
                cos_dist[i] = 0
        normal_acc = np.mean((cos_dist==label).astype(int))
            
        return best_acc, best_th,normal_acc
    
    def test_distance(self, testloader):
        self.net.eval()
        device = self.device
        distance_ls = torch.tensor([])
        label_ls = torch.tensor([])
        with torch.no_grad():
            for data in testloader:
                x1, x2, label = data
                x1, x2, label = x1.to(device), x2.to(device), label
                cos_dist = self.cal_similarity(x1, x2).cpu()
                distance_ls = torch.cat([distance_ls, cos_dist], dim=0)
                label_ls = torch.cat([label_ls, label], dim=0)
        
        distance_ls = distance_ls.numpy()
        label_ls = label_ls.numpy()
        best_acc, best_th,normal_acc = self.cal_accuracy(distance_ls, label_ls)
        return best_acc, best_th,normal_acc
    
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

            self.opt.zero_grad()
            # print(label)
            out = self.identification(x1, x2)
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
        distance_ls = []
        with torch.no_grad():
            for data in testloader:
                x1, x2, label = data
                x1, x2, label = x1.to(device), x2.to(device), label.to(device)
                # x1 = x1.to(torch.float)
                # x2 = x2.to(torch.float)
                out = self.identification(x1, x2)
                _, pred = torch.max(out.data, 1)
                total += label.size(0)
                correct += (pred == label).sum().item()
        acc = correct / total
        return acc
       
    def infer(self, data):
        pass   




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
               
class Video_supervise_Net_ms1m(nn.Module):
    def __init__(self, loss_type, GPU_ID, num_class, dim,test=False):
        ################## for the design of the network: the embed_dim should divide the num_heads
        super().__init__()

        self.net = SwinTransformer3D(embed_dim=48, 
                                  depths=[2, 2, 6, 2],
                                  num_heads=[3, 6, 12, 24],
                                  patch_size=(2, 4, 4),
                                  window_size=(8, 7, 7),
                                  drop_path_rate=0.1,
                                  patch_norm=True)

        
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
        )
        self.loss_type = loss_type
        self.GPU_ID = GPU_ID

        self.loss = CosFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID)

        
    
    def forward(self, img, label= None):
        x = self.net(img)
        x = x.mean(dim=[2, 3, 4])
        x = self.to_latent(x)
        emb = self.mlp_head(x)
        if label is not None:
            x = self.loss(emb, label)
            return x, emb
        else:
            return emb    


class Video_supervise_Net_ms1m_pretrained(nn.Module):
    def __init__(self, loss_type, GPU_ID, num_class, dim,test=False):
        ################## for the design of the network: the embed_dim should divide the num_heads
        super().__init__()

        self.net = SwinTransformer3D(embed_dim=48, 
                                  depths=[2, 2, 6, 2],
                                  num_heads=[3, 6, 12, 24],
                                  patch_size=(2, 4, 4),
                                  window_size=(8, 7, 7),
                                  drop_path_rate=0.1,
                                  patch_norm=True)

        print('load pretrained model ', './results/3DSWIN-ms1m/Backbone_ms1m_vi_Epoch_21_Batch_102660_Time_2022-10-12-16-33_checkpoint.pth')
        checkpoint = torch.load('./results/3DSWIN-ms1m/Backbone_ms1m_vi_Epoch_21_Batch_102660_Time_2022-10-12-16-33_checkpoint.pth')
        
        model_dict = self.net.state_dict()
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items(): 
            name = k[4:]
            if name in model_dict:
                new_state_dict[name] = v

        
        self.net.load_state_dict(new_state_dict)
        
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
        )
        self.loss_type = loss_type
        self.GPU_ID = GPU_ID

        self.loss = CosFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID)

        
    
    def forward(self, img, label= None):
        x = self.net(img)
        x = x.mean(dim=[2, 3, 4])
        x = self.to_latent(x)
        emb = self.mlp_head(x)
        if label is not None:
            x = self.loss(emb, label)
            return x, emb
        else:
            return emb    

class Vit_supervise_Net_ms1m(nn.Module):
    def __init__(self, loss_type, GPU_ID, num_class, dim,test=False):
        ################## for the design of the network: the embed_dim should divide the num_heads
        super().__init__()

        self.net = SwinTransformer3D(embed_dim=48, 
                                  depths=[2, 2, 6, 2],
                                  num_heads=[3, 6, 12, 24],
                                  patch_size=(2, 4, 4),
                                  window_size=(8, 7, 7),
                                  drop_path_rate=0.1,
                                  patch_norm=True)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
        )
        self.loss_type = loss_type
        self.GPU_ID = GPU_ID

        self.loss = CosFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID)

        
    
    def forward(self, img, label= None):
        x = self.net(img)
        x = x.mean(dim=[2, 3, 4])
        x = self.to_latent(x)
        emb = self.mlp_head(x)
        if label is not None:
            x = self.loss(emb, label)
            return x, emb
        else:
            return emb 

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