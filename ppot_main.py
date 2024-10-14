import os,ot
from torch.utils.data import Dataset, DataLoader
import scipy.io as io
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os.path as osp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import manifold
# import copy
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import torch.nn.utils.weight_norm as weightNorm
import torch.utils.data
from ot.da import sinkhorn_l1l2_gl
from utils import computeTransportSinkhorn, sinkhorn_R1reg,sinkhorn_R1reg_lab
import csv
device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')


def get_mat_path(dset, domain):
    if dset =='imageCLEF':
        data_dir = '/resnet50_clef'
        if domain == 'c':
            file_name = 'c.mat'
        elif domain == 'i':
            file_name = 'i.mat'
        elif domain == 'p':
            file_name = 'p.mat'
            
    if dset == 'Office31':
        data_dir = '/Office31'
        if domain == 'amazon':
            file_name = 'office-A-resnet50.mat'
        elif domain == 'webcam':
            file_name = 'office-W-resnet50.mat'
        elif domain == 'dslr':
            file_name = 'office-D-resnet50.mat'

    path = os.path.join(data_dir, file_name)
    return path

def  Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

class ImageSet_dataset(Dataset):
    def __init__(self, mat_path):
        data = io.loadmat(mat_path)
        img = torch.from_numpy(data['resnet50_features'])
        img = img.view([img.shape[0], -1])
        label = torch.from_numpy(data['labels'].squeeze(0))
        self.img = img
        self.label = label

    def __getitem__(self, idx):
        Batch_data = self.img[idx, :]
        Batch_label = self.label[idx]
        return Batch_data, Batch_label

    def __len__(self):
        return len(self.label)

def Make_Loader(mat_path, stage, domain, Batch_size=None,train_batch_sampler=None):
    data_set = ImageSet_dataset(mat_path)
    
    if Batch_size is None:
        Batch_size = len(data_set)
    if stage == 'train':
        if domain == 'source':
            # new_loader = DataLoader(data_set, batch_size=Batch_size, num_workers=0, shuffle=True, drop_last=True)
            new_loader = DataLoader(data_set, batch_sampler=train_batch_sampler, num_workers=0)
        else:
            new_loader = DataLoader(data_set, batch_size=Batch_size, num_workers=0, shuffle=True, drop_last=True)
    else:
        new_loader = DataLoader(data_set, Batch_size, num_workers=0, shuffle=False)
    return new_loader

class InfiniteSliceIterator:
    def __init__(self, array, class_):
        assert type(array) is np.ndarray
        self.array = array
        self.i = 0
        self.class_ = class_

    def reset(self):
        self.i = 0

    def get(self, n):
        len_ = len(self.array)
        # not enough element in 'array'
        if len_ < n:
            print(f"there are really few items in class {self.class_}")
            self.reset()
            np.random.shuffle(self.array)
            mul = n // len_
            rest = n - mul * len_
            return np.concatenate((np.tile(self.array, mul), self.array[:rest]))

        # not enough element in array's tail
        if len_ - self.i < n:
            self.reset()

        if self.i == 0:
            np.random.shuffle(self.array)
        i = self.i
        self.i += n
        return self.array[i : self.i]
    
class BalancedBatchSampler(torch.utils.data.sampler.BatchSampler):
    def __init__(self, labels, batch_size):
        classes = sorted(set(labels.numpy()))
        # print(classes)
    
        n_classes = len(classes)
        self._n_samples = batch_size // n_classes
        if self._n_samples == 0:
            raise ValueError(
                f"batch_size should be bigger than the number of classes, got {batch_size}"
            )
    
        self._class_iters = [
            InfiniteSliceIterator(np.where(labels == class_)[0], class_=class_)
            for class_ in classes
        ]
    
        batch_size = self._n_samples * n_classes
        self.n_dataset = len(labels)
        self._n_batches = self.n_dataset // batch_size
        if self._n_batches == 0:
            raise ValueError(
                f"Dataset is not big enough to generate batches with size {batch_size}"
            )
        # print("K=", n_classes, "nk=", self._n_samples)
        # print("Batch size = ", batch_size)
    
    def __iter__(self):
        for _ in range(self._n_batches):
            indices = []
            for class_iter in self._class_iters:
                indices.extend(class_iter.get(self._n_samples))
            np.random.shuffle(indices)
            yield indices
    
        for class_iter in self._class_iters:
            class_iter.reset()
    
    def __len__(self):
        return self._n_batches
    
def dist(X, Y):
    xx = X.pow(2).sum(1).repeat(Y.shape[0], 1)
    xy = X @ Y.t()
    yy = Y.pow(2).sum(1).repeat(X.shape[0], 1)
    dist = xx.t() + yy - 2 * xy
    return dist

def normal_dist(X, Y):
    norm_X = X.pow(2).sum(1).pow(0.5).detach().unsqueeze(1)
    norm_Y = Y.pow(2).sum(1).pow(0.5).detach().unsqueeze(1)
    
    X = X * (1/norm_X)
    Y = Y * (1/norm_Y)
    xx = X.pow(2).sum(1).repeat(Y.shape[0], 1)
    xy = X @ Y.t()
    yy = Y.pow(2).sum(1).repeat(X.shape[0], 1)
    dist = xx.t() + yy - 2 * xy
    return dist

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        pred_value, pred = output.topk(maxk, 1, True, True)# pred是标签索引，不是softmax的值。
        pred = pred.t()
        correct = pred.eq(target[None])
        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class feat_classifier(nn.Module):
    def __init__(self, class_num=31, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)
    def forward(self, x):
        x = self.fc(x)
        return x

class FC_layers(nn.Module):
    def __init__(self, conv_dim_1, conv_dim_2):
        super(FC_layers, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(2048, conv_dim_1),
            nn.BatchNorm1d(conv_dim_1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(conv_dim_1, conv_dim_2),
            nn.BatchNorm1d(conv_dim_2),
            nn.Tanh(),
        )

        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Dataclassifier(nn.Module):
    def __init__(self, conv_dim_1, n_class):
        super(Dataclassifier, self).__init__()
        self.fc1 = nn.Linear(conv_dim_1, conv_dim_1)
        self.fc2 = nn.Linear(conv_dim_1, n_class)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        return x

class ForeverDataIterator:
    """A data iterator that will never stop producing data"""
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.FloatTensor).mean()
    else:
        return (pred == label).type(torch.FloatTensor).mean()
    
def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad
        
def evaluate_data_classifier(source_domain, target_domain, feat_extract,data_classifier,data_loader,domain=None):
    feat_extract.eval()
    data_classifier.eval()
    correct = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(data_loader):
            images, target = images.to(device), target.to(device)
            output = data_classifier(feat_extract(images))
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()
        accur = correct.item() / len(data_loader.dataset)
    if domain =='target':
        print('Task: {}/{}, Accuracy: {}/{} ({:.1f}%)'.format(
        source_domain, target_domain, correct, len(data_loader.dataset), 100 * accur))
    return accur

def extract_prototypes(loader, Feat_extractor, n_clusters):
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            data = Feat_extractor(data).to(device).cpu().numpy()
            target = target.cpu().numpy()
            if i == 0:
                X = data
                y = target
            else:
                
                X = np.vstack((X, data))
                y = np.hstack((y, target))
        n_hidden = X.shape[1]
        mean_mat = np.zeros((n_clusters, n_hidden))
        number_in_class = np.zeros(n_clusters)
        for i in range(n_clusters):
            mean_mat[i] = np.mean(X[y==i,:],axis=0)
            number_in_class[i] = np.sum(y==i)
        return mean_mat, X
    
def obtain_label(loader, netF, netB):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(device)
            feas = netF(inputs)
            outputs = netB(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    # if args.distance == 'cosine':
    #     all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    #     all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    all_output = torch.softmax(all_output / 1,dim=1)
   
    aff = all_output.float().cpu().numpy()
    
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>0)
    labelset = labelset[0]

    
    dd = ot.dist(all_fea, initc[labelset])
    # dd = np.matmul(all_fea, initc[labelset].T)
    # pred_label = dd.argmax(axis=1)
    pred_label = dd.argmin(axis=1)

    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = ot.dist(all_fea, initc[labelset])
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    print(log_str+'\n')

    return initc, pred_label

def obtain_totalabel(fea,all_label,netB):
    with torch.no_grad():
            all_fea = fea
            all_fea = all_fea.to(device)
            all_output = netB(all_fea)
    _, predict = torch.max(all_output, 1)
    all_label = torch.from_numpy(all_label).to(device)
  
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    # if args.distance == 'cosine':
    #     all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    #     all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    all_output = torch.softmax(all_output / 1,dim=1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    cls_count = np.eye(K)[predict.cpu()].sum(axis=0)
    labelset = np.where(cls_count>0)
    labelset = labelset[0]
    # print(labelset)
    dd = ot.dist(all_fea, initc[labelset])
    # dd = np.matmul(all_fea, initc[labelset].T)
    # pred_label = dd.argmax(axis=1)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = ot.dist(all_fea, initc[labelset])
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().cpu().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    print(log_str+'\n')

    return initc, pred_label

def loop_iterable(iterable):
    while True:
        yield from iterable
        
def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        
    return optimizer

def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=100,lr_decay_factor=0.5):
    init_lr = optimizer.param_groups[0]['lr']
    if epoch > 0 and (epoch % lr_decay_epoch == 0):
        lr = init_lr*lr_decay_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return optimizer

    

save_name = ('AtoW', 'AtoD', 'WtoA', 'WtoD', 'DtoA', 'DtoW')
source_domain_set = ( 'amazon', 'webcam', 'webcam', 'dslr', 'dslr')
target_domain_set = ('webcam', 'dslr', 'amazon', 'dslr', 'amazon', 'webcam')

dset = 'Office31'

n_class = 12
epochs = 100
hidden_size = 256
weight_decay = 1e-3
# class_num=31
temprature = 1.0
reg = 1e1
ave=[]
eta = 1e2

batch_size = 64
trade_off = 1e-3
trade_off1 = 1e1
interval_label = 5
init_lr = 1e-2
lr_weight = 1.0
epoch_start_align = 50
Nce_loss = False
result = np.zeros([len(save_name),10])


for Domain_iter in range(len(save_name)):
            for experiment in range(10):
                    tot_loss = []
                    epoch_iter = []
                    target_acc = []
                    source_acc = []
                    best_acc = 0
                    total_acc = 0
                    best_summap = 0
                    Domain_iter = 0
                    source_domain = source_domain_set[Domain_iter]
                    target_domain = target_domain_set[Domain_iter]
                    source_path = get_mat_path(dset, source_domain)
                    target_path = get_mat_path(dset, target_domain)
                    # source_loader = Make_Loader(source_path, 'train', Batch_size=batch_size)
                    data = io.loadmat(source_path)
                    source_labels = torch.from_numpy(data['labels'].squeeze(0))
                    train_batch_sampler = BalancedBatchSampler(source_labels, batch_size=batch_size)
                    source_loader = Make_Loader(source_path, 'train', 'source',train_batch_sampler=train_batch_sampler)
                    # source_loader = Make_Loader(source_path, 'train', 'source',Batch_size=batch_size)
                    target_loader = Make_Loader(target_path, 'train', 'target',Batch_size=batch_size)
                    iter_distance = len(source_loader) 
                    test_loader = Make_Loader(target_path, 'test', 'target')
            
    
                    feat_extractor = FC_layers(conv_dim_1=1024, conv_dim_2=hidden_size).to(device)
                    feat_extractor.apply(init_weights)
                    extract_total = FC_layers(conv_dim_1=1024, conv_dim_2=hidden_size).to(device)
                    extract_total.apply(init_weights)
                    data_classifier = feat_classifier(bottleneck_dim=hidden_size, class_num=n_class).to(device)
                    data_classifier.apply(init_weights)
                    data_classifier_w = feat_classifier(bottleneck_dim=hidden_size, class_num=n_class).to(device)
                    data_classifier_w.apply(init_weights)
                    classifier_total = feat_classifier(bottleneck_dim=hidden_size, class_num=n_class).to(device)
                    classifier_total.apply(init_weights)
                    
                    optimizer_feat_extractor = optim.SGD(feat_extractor.parameters(), lr=init_lr, weight_decay=1e-3)
                    optimizer_data_classifier = optim.SGD(data_classifier.parameters(), lr=init_lr, weight_decay=1e-3)
                    optimizer_data_classifier_w = optim.SGD(data_classifier_w.parameters(), lr=init_lr, weight_decay=1e-3)
                    ####################
                    # Train procedure
                    ####################
                    Map_ave = []
                    for epoch in range(epochs):
                        if epoch == epoch_start_align:
                              data_classifier_w.load_state_dict(data_classifier.state_dict())
                        
                        if epoch % interval_label == 0 and epoch >= epoch_start_align:
                            set_requires_grad(feat_extractor,requires_grad=False)
                            set_requires_grad(data_classifier_w,requires_grad=False)
                            init_centroid, _ = obtain_label(test_loader, feat_extractor, data_classifier_w)
                    
                        S_batches = loop_iterable(source_loader)
                        batch_iterator = zip(S_batches, loop_iterable(target_loader))
                        iterations = len(source_loader)
                        
                        
                        wass_loss_tot,clf_loss,total_loss,nce_sloss,clf_sloss = 0,0,0,0,0
                        
                        for batch_idx in range(iterations):
                            # print('iterations is {}'.format(i))
                            (X_s, lab_s), (X_t, lab_t) = next(batch_iterator)
                            X_s, lab_s = X_s.to(device), lab_s.to(device)
                            X_t, lab_t = X_t.to(device), lab_t.to(device)
                         
                            if epoch > epoch_start_align:
                                #######  Optimal Transport calculation #######
                                ##############################################
                                p = (batch_idx + (epoch - epoch_start_align) * len(source_loader)) / (
                                    len(source_loader) * (epochs - epoch_start_align))
                                lr = float(init_lr / (1. + 10 * p) ** 0.75)
                                set_lr(optimizer_feat_extractor, lr * lr_weight)
                                set_lr(optimizer_data_classifier_w , lr * lr_weight)
                                exp_lr_scheduler(optimizer_feat_extractor, epoch, lr_decay_epoch=100,lr_decay_factor=0.5)
                                exp_lr_scheduler(optimizer_data_classifier_w,epoch, lr_decay_epoch=100,lr_decay_factor=0.5)
                                set_requires_grad(feat_extractor, requires_grad=True)
                                set_requires_grad(data_classifier, requires_grad=False)
                                set_requires_grad(data_classifier_w, requires_grad=True)
                                
                                z = feat_extractor(torch.cat((X_s, X_t), 0))
                                z_s, z_t = z[:X_s.shape[0]], z[X_s.shape[0]:]
                                
                                
                                dis_preds =  ot.dist(z_t.detach().cpu().numpy(),init_centroid)
                               
                                preds = dis_preds.argmin(axis=1)
                                # sum_acc = torch.sum(torch.from_numpy(preds) == lab_t.cpu()).float() / z_t.shape[0]
                                # print(sum_acc)
                                # acc = np.sum(preds == lab_t.cpu().numpy()) / len(lab_t)
                                
                                # estimate target data class propotion.
                                propotionT = torch.zeros(class_num)
                                for i in range(class_num):
                                    propotionT[i] = torch.sum(preds==i)
                                
                                #propotion_final = propotionT / torch.sum(propotionT)
                                
                                dis_preds = torch.from_numpy(dis_preds)
                                preds = torch.from_numpy(preds)
                                aux_intra = batch_size * np.ones((z_s.shape[0],z_t.shape[0]))
                                aux_intra[lab_s.cpu().unsqueeze(1)==preds.unsqueeze(0)]=1
                                
                                aux_inter = 1 * np.ones((z_s.shape[0], z_t.shape[0]))
                                aux_inter[lab_s.cpu().unsqueeze(1)==preds.unsqueeze(0)]=1e-5

                                epsilon = 1e-5
                                inter = epsilon / X_s.shape[0] / (X_t.shape[0] * torch.max(propotionT))
                                intra = 1 / X_s.shape[0] / torch.min(propotionT)
                                
                               
                                distance = dist(z_s, z_t)
                                distance = distance.to(device)
                                
                                # z_s = F.normalize(z_s,dim=1)
                                # z_t = F.normalize(z_t,dim=1)
    
                                phi = sinkhorn_R1reg_lab(np.ones(X_s.shape[0])/X_s.shape[0], np.ones(X_t.shape[0])/X_t.shape[0],
                                      distance.detach().cpu().numpy(), reg, eta=eta, numItermax=5,numInnerItermax=5,
                                      intra_class=intra, inter_class=inter,aux=aux_intra,aux1=aux_inter)
                                
                                
                               
                           
                                phi = torch.from_numpy(phi).detach().to(device)
                                ot_loss = (phi * distance).sum()
                                
                                source_preds_s = data_classifier(z_s)
                                criterion = nn.CrossEntropyLoss().to(device)
                                clf_s_loss = F.cross_entropy(source_preds_s, lab_s.long())
                                
                                z_s_hat = torch.mm(phi.float(), z_t.float()) * z_s.shape[0]
                                
                                
                                source_preds = data_classifier_w(z_s_hat)
                                clf_loss = criterion(source_preds, lab_s.long())
                                target_preds = data_classifier_w(z_t)
                                
                                
                                loss = clf_s_loss + trade_off * ot_loss + trade_off1 * clf_loss #+ trade_off2 * nce_loss
                           
                                optimizer_feat_extractor.zero_grad()
                                optimizer_data_classifier.zero_grad()
                                optimizer_data_classifier_w.zero_grad()
                                loss.backward()
                                optimizer_feat_extractor.step()
                                optimizer_data_classifier.step()
                                optimizer_data_classifier_w.step()
                                
                                wass_loss_tot += ot_loss.item()
                                
                                #######  Train target classifier #######
                                ##############################################
                            else:
                                set_requires_grad(data_classifier, requires_grad=True)
                                set_requires_grad(feat_extractor, requires_grad=True)
                                z = feat_extractor(torch.cat((X_s, X_t), 0))
                                
                                source_preds = data_classifier(z[:X_s.shape[0]])
                                criterion = nn.CrossEntropyLoss()
                                clf_loss = criterion(source_preds, lab_s.long())
                                loss = clf_loss
                                
                                optimizer_feat_extractor.zero_grad()
                                optimizer_data_classifier.zero_grad()
                                loss.backward()
                                optimizer_feat_extractor.step()
                                optimizer_data_classifier.step()
                        

                        total_loss += loss.item()
                        tot_loss.append(total_loss)
                        clf_sloss += clf_loss.item()
                        # if epoch == 50:
                        #     feat_extractor.eval()
                        #     _,_,_,_ =tsne_feature(Domain_iter,feat_extractor,'pretrain',trade,trade_off1,batch_size)
                        if epoch % 5 == 0:
                            print('\OT_test Train Epoch:{} \t  total_loss:{:.4f} \t  target_loss:{:.4f} \t  ot_loss:{:.4f}'.format(epoch,total_loss,clf_sloss,wass_loss_tot))
                        if epoch % 1 == 0:
                            if epoch < epoch_start_align:
                                acc = evaluate_data_classifier(source_domain, target_domain, feat_extractor, data_classifier, test_loader,'target')
                                # acc_s = evaluate_data_classifier(source_domain, target_domain, feat_extractor, data_classifier, source_loader)
                                epoch_iter.append(epoch)
                                # source_acc.append(acc_s)
                                target_acc.append(acc)
                            else:
                                acc = evaluate_data_classifier(source_domain, target_domain, feat_extractor, data_classifier_w, test_loader,'target')
                                # acc_s = evaluate_data_classifier(source_domain, target_domain, feat_extractor, data_classifier, source_loader)
                                epoch_iter.append(epoch)
                                # source_acc.append(acc_s)
                                target_acc.append(acc)
                            if acc > best_acc:
                                best_acc = acc
                                extract_total.load_state_dict(feat_extractor.state_dict())
                                classifier_total.load_state_dict(data_classifier_w.state_dict())
                                
                    Map_ave = np.array(Map_ave).mean()
                    ave.append(Map_ave)
                    result[Domain_iter][experiment] = best_acc
                    print('Task: {}/{} best_acc = {:.3f}'.format(source_domain, target_domain, best_acc)) 
                    
                
