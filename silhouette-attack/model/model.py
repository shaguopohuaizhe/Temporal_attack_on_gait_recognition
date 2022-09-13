import math
import os
import os.path as osp
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as tordata

from .network import TripletLoss, SetNet
from .utils import TripletSampler

from attacks import LinfMomentumIterativeAttack
#from advertorch.utils import normalize_by_pnorm,clamp
from torchvision import transforms
import cv2
import torch.nn.functional as F

from wgan import GoodGenerator, GoodDiscriminator
from collections import OrderedDict

class attack_cosine_distance(nn.CosineEmbeddingLoss):
    def __init__(self, target, margin=0., size_average=None, reduce=None, reduction='mean'):
        super().__init__()#margin, size_average, reduce, reduction
        self.target = target
    
    def forward(self, input1, input2):
        return super().forward(input1, input2, target=self.target)
            
class Model:
    def __init__(self,
                 hidden_dim,
                 lr,
                 hard_or_full_trip,
                 margin,
                 num_workers,
                 batch_size,
                 restore_iter,
                 total_iter,
                 save_name,
                 train_pid_num,
                 frame_num,
                 model_name,
                 train_source,
                 test_source,
                 img_size=64):

        self.save_name = save_name
        self.train_pid_num = train_pid_num
        self.train_source = train_source
        self.test_source = test_source

        self.hidden_dim = hidden_dim
        self.lr = lr
        self.hard_or_full_trip = hard_or_full_trip
        self.margin = margin
        self.frame_num = frame_num
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.model_name = model_name
        self.P, self.M = batch_size

        self.restore_iter = restore_iter
        self.total_iter = total_iter

        self.img_size = img_size

        self.encoder = SetNet(self.hidden_dim).float()
        self.encoder = nn.DataParallel(self.encoder)
        self.triplet_loss = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin).float()
        self.triplet_loss = nn.DataParallel(self.triplet_loss)
        self.encoder.cuda()
        self.triplet_loss.cuda()

        self.optimizer = optim.Adam([
            {'params': self.encoder.parameters()},
        ], lr=self.lr)

        self.hard_loss_metric = []
        self.full_loss_metric = []
        self.full_loss_num = []
        self.dist_list = []
        self.mean_dist = 0.01

        self.sample_type = 'all'

    def collate_fn(self, batch):
        batch_size = len(batch)
        feature_num = len(batch[0][0])
        seqs = [batch[i][0] for i in range(batch_size)]
        frame_sets = [batch[i][1] for i in range(batch_size)]
        view = [batch[i][2] for i in range(batch_size)]
        seq_type = [batch[i][3] for i in range(batch_size)]
        label = [batch[i][4] for i in range(batch_size)]
        batch = [seqs, view, seq_type, label, None]

        def select_frame(index):
            sample = seqs[index]
            frame_set = frame_sets[index]
            if self.sample_type == 'random':
                frame_id_list = random.choices(frame_set, k=self.frame_num)
                _ = [feature.loc[frame_id_list].values for feature in sample]
            else:
                _ = [feature.values for feature in sample]
            return _

        seqs = list(map(select_frame, range(len(seqs))))

        if self.sample_type == 'random':
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
        else:
            gpu_num = min(torch.cuda.device_count(), batch_size)
            batch_per_gpu = math.ceil(batch_size / gpu_num)
            batch_frames = [[
                                len(frame_sets[i])
                                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                if i < batch_size
                                ] for _ in range(gpu_num)]
            if len(batch_frames[-1]) != batch_per_gpu:
                for _ in range(batch_per_gpu - len(batch_frames[-1])):
                    batch_frames[-1].append(0)
            max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])
            seqs = [[
                        np.concatenate([
                                           seqs[i][j]
                                           for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                           if i < batch_size
                                           ], 0) for _ in range(gpu_num)]
                    for j in range(feature_num)]
            seqs = [np.asarray([
                                   np.pad(seqs[j][_],
                                          ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
                                          'constant',
                                          constant_values=0)
                                   for _ in range(gpu_num)])
                    for j in range(feature_num)]
            batch[4] = np.asarray(batch_frames)

        batch[0] = seqs
        return batch
        
    def collate_fn_A(self, batch):
        batch_size = len(batch)
        feature_num = len(batch[0][0])
        seqs = [batch[i][0] for i in range(batch_size)]
        frame_sets = [batch[i][1] for i in range(batch_size)]
        view = [batch[i][2] for i in range(batch_size)]
        label = [batch[i][3] for i in range(batch_size)]
        batch = [seqs, view, label, None]

        def select_frame(index):
            sample = seqs[index]
            frame_set = frame_sets[index]
            if self.sample_type == 'random':
                frame_id_list = random.choices(frame_set, k=self.frame_num)
                _ = [feature.loc[frame_id_list].values for feature in sample]
            else:
                _ = [feature.values for feature in sample]
            return _

        seqs = list(map(select_frame, range(len(seqs))))

        if self.sample_type == 'random':
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
        else:
            gpu_num = min(torch.cuda.device_count(), batch_size)
            batch_per_gpu = math.ceil(batch_size / gpu_num)
            batch_frames = [[
                                len(frame_sets[i])
                                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                if i < batch_size
                                ] for _ in range(gpu_num)]
            if len(batch_frames[-1]) != batch_per_gpu:
                for _ in range(batch_per_gpu - len(batch_frames[-1])):
                    batch_frames[-1].append(0)
            max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])
            seqs = [[
                        np.concatenate([
                                           seqs[i][j]
                                           for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                           if i < batch_size
                                           ], 0) for _ in range(gpu_num)]
                    for j in range(feature_num)]
            seqs = [np.asarray([
                                   np.pad(seqs[j][_],
                                          ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
                                          'constant',
                                          constant_values=0)
                                   for _ in range(gpu_num)])
                    for j in range(feature_num)]
            batch[3] = np.asarray(batch_frames)

        batch[0] = seqs
        return batch

    def fit(self):
        if self.restore_iter != 0:
            self.load(self.restore_iter)

        self.encoder.train()
        self.sample_type = 'random'
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        triplet_sampler = TripletSampler(self.train_source, self.batch_size)
        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_sampler=triplet_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        train_label_set = list(self.train_source.label_set)
        train_label_set.sort()

        _time1 = datetime.now()
        for seq, view, seq_type, label, batch_frame in train_loader:
            self.restore_iter += 1
            self.optimizer.zero_grad()

            for i in range(len(seq)):
                seq[i] = self.np2var(seq[i]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()

            feature, label_prob = self.encoder(*seq, batch_frame)

            target_label = [train_label_set.index(l) for l in label]
            target_label = self.np2var(np.array(target_label)).long()

            triplet_feature = feature.permute(1, 0, 2).contiguous()
            triplet_label = target_label.unsqueeze(0).repeat(triplet_feature.size(0), 1)
            (full_loss_metric, hard_loss_metric, mean_dist, full_loss_num
             ) = self.triplet_loss(triplet_feature, triplet_label)
            if self.hard_or_full_trip == 'hard':
                loss = hard_loss_metric.mean()
            elif self.hard_or_full_trip == 'full':
                loss = full_loss_metric.mean()

            self.hard_loss_metric.append(hard_loss_metric.mean().data.cpu().numpy())
            self.full_loss_metric.append(full_loss_metric.mean().data.cpu().numpy())
            self.full_loss_num.append(full_loss_num.mean().data.cpu().numpy())
            self.dist_list.append(mean_dist.mean().data.cpu().numpy())

            if loss > 1e-9:
                loss.backward()
                self.optimizer.step()

            if self.restore_iter % 1000 == 0:
                print(datetime.now() - _time1)
                _time1 = datetime.now()

            if self.restore_iter % 100 == 0:
                self.save()
                print('iter {}:'.format(self.restore_iter), end='')
                print(', hard_loss_metric={0:.8f}'.format(np.mean(self.hard_loss_metric)), end='')
                print(', full_loss_metric={0:.8f}'.format(np.mean(self.full_loss_metric)), end='')
                print(', full_loss_num={0:.8f}'.format(np.mean(self.full_loss_num)), end='')
                self.mean_dist = np.mean(self.dist_list)
                print(', mean_dist={0:.8f}'.format(self.mean_dist), end='')
                print(', lr=%f' % self.optimizer.param_groups[0]['lr'], end='')
                print(', hard or full=%r' % self.hard_or_full_trip)
                sys.stdout.flush()
                self.hard_loss_metric = []
                self.full_loss_metric = []
                self.full_loss_num = []
                self.dist_list = []

            # Visualization using t-SNE
            # if self.restore_iter % 500 == 0:
            #     pca = TSNE(2)
            #     pca_feature = pca.fit_transform(feature.view(feature.size(0), -1).data.cpu().numpy())
            #     for i in range(self.P):
            #         plt.scatter(pca_feature[self.M * i:self.M * (i + 1), 0],
            #                     pca_feature[self.M * i:self.M * (i + 1), 1], label=label[self.M * i])
            #
            #     plt.show()

            if self.restore_iter == self.total_iter:
                break

    def ts2var(self, x):
        return autograd.Variable(x).cuda()

    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x))

    def transform(self, flag, batch_size=1):
        self.encoder.eval()
        source = self.test_source if flag == 'test' else self.train_source
        self.sample_type = 'all'
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        feature_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()
        #seq_temp = [np.random.uniform(size=(1,1,64,44))]

        for i, x in enumerate(data_loader):
            #print(x)
            seq, view, seq_type, label, batch_frame = x
            '''gallery_seq_dict = [['nm-01'], ['nm-02'], ['nm-03'], ['nm-04']]
            if seq_type not in gallery_seq_dict:
                num = int(batch_frame / 10)+1
                #print(num)
                seq[0][:,:num,:,:] = np.where(seq[0][:,:num,:,:] > 0.5, 1, seq[0][:,:num,:,:])
                seq[0][:,:num,:,:] = np.where(seq[0][:,:num,:,:] <= 0.5, 0, seq[0][:,:num,:,:])'''
                    
            #print(seq[0].shape)
            #print(np.random.uniform(size=(1,1,64,44)).shape)
            #seq[0] = np.concatenate((seq[0],seq_temp[0][:,0:50,:,:]),axis=1)
            #seq_temp = seq
            #seq[0] = np.concatenate((seq[0],np.random.uniform(size=(1,100,64,44))),axis=1)
            
            '''if i == 0:
                temp = seq[0][:,:1,:,:]
            else:
                gallery_seq_dict = [['nm-01'], ['nm-02'], ['nm-03'], ['nm-04']]
                if seq_type not in gallery_seq_dict:
                    #seq[0] = np.delete(seq[0],np.s_[0:-20],1)
                    print(seq[0])
                    print(temp)
                    seq[0] = np.concatenate((seq[0],temp),axis=1)
                    batch_frame += 1'''
                
            
            '''gallery_seq_dict = [['nm-01'], ['nm-02'], ['nm-03'], ['nm-04']]
            if seq_type not in gallery_seq_dict:
                #seq[0] = np.delete(seq[0],np.s_[0:-20],1)
                seq[0] = np.concatenate((seq[0],np.random.uniform(size=(1,1,64,44))),axis=1)
                batch_frame += 1'''
                
            for j in range(len(seq)):
                seq[j] = self.np2var(seq[j]).float()
            #print(seq[0].shape)
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()
                #print(batch_frame)#, np.sum(batch_frame))

            feature, _ = self.encoder(*seq, batch_frame)
            n, num_bin, _ = feature.size()
            #print(n)
            feature_list.append(feature.view(n, -1).data.cpu().numpy())
            view_list += view
            seq_type_list += seq_type
            label_list += label

        return np.concatenate(feature_list, 0), view_list, seq_type_list, label_list
        
    def transform_A(self, flag, batch_size=1):
        self.encoder.eval()
        source = self.test_source if flag == 'test' else self.train_source
        self.sample_type = 'all'
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn_A,
            num_workers=self.num_workers)

        feature_list = list()
        view_list = list()
        label_list = list()
        #seq_temp = [np.random.uniform(size=(1,1,64,44))]

        for i, x in enumerate(data_loader):
            seq, view, label, batch_frame = x
                
            for j in range(len(seq)):
                seq[j] = self.np2var(seq[j]).float()
            #print(seq[0].shape)
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()
                #print(batch_frame)#, np.sum(batch_frame))

            feature, _ = self.encoder(*seq, batch_frame)
            n, num_bin, _ = feature.size()
            #print(n)
            feature_list.append(feature.view(n, -1).data.cpu().numpy())
            view_list += view
            label_list += label

        return np.concatenate(feature_list, 0), view_list, label_list
        
    def attack_pgd(self, flag, batch_size=1):
        self.encoder.eval()
        source = self.test_source if flag == 'test' else self.train_source
        self.sample_type = 'all'
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        feature_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()

        for i, x in enumerate(data_loader):
            seq, view, seq_type, label, batch_frame = x
            gallery_seq_dict = [['nm-01'], ['nm-02'], ['nm-03'], ['nm-04']]
            if seq_type not in gallery_seq_dict:
                
                for j in range(len(seq)):
                    seq[j] = self.np2var(seq[j]).float()
                if batch_frame is not None:
                    batch_frame = self.np2var(batch_frame).int()
                # print(batch_frame, np.sum(batch_frame))
                
                feature, _ = self.encoder(*seq, batch_frame)
                
                '''transform = trans.Compose([
                        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    ])'''
                #print(*seq)
                #print(seq[0][0,0,0,:])
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                adversary = LinfMomentumIterativeAttack(
                            self.encoder, loss_fn=attack_cosine_distance(target=torch.ones(1).to(device)), eps=0.1,
                            nb_iter=20, eps_iter=0.01, decay_factor=1., clip_min=0.0, clip_max=1.0,
                            targeted=False)
                adv = adversary.perturb(*seq, batch_frame, feature)
            
                pad_top = 0
                pad_bottom = 0
                pad_left = 10
                pad_right = 10
                adv = F.pad(adv, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

                #adv = self.perturb(*seq, batch_frame, feature)
                for j in range(adv.shape[1]):
                    img_mask=adv[0][j].cpu().numpy()*255.0
                    label_path=os.path.join("/GaitSet/output_pgd/", label[0])
                    if not os.path.exists(label_path):
                        os.mkdir(label_path)
                    seq_type_path = os.path.join(label_path, seq_type[0])
                    if not os.path.exists(seq_type_path):
                        os.mkdir(seq_type_path)
                    view_path = os.path.join(seq_type_path, view[0])
                    if not os.path.exists(view_path):
                        os.mkdir(view_path)
                    out_path = os.path.join(view_path, label[0]+'-'+seq_type[0]+'-'+view[0]+'-'+'{:0>3d}.png'.format(j+1))
                    
                    print(out_path)
                    cv2.imwrite(out_path,img_mask)
                    
                    
    def attack(self, flag, batch_size=1):
        self.encoder.eval()
        source = self.test_source if flag == 'test' else self.train_source
        self.sample_type = 'all'
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        feature_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()

        for i, x in enumerate(data_loader):
            path = "/GaitSet/output/"
            seq, view, seq_type, label, batch_frame = x
            gallery_seq_dict = [['nm-01'], ['nm-02'], ['nm-03'], ['nm-04']]
            if seq_type in gallery_seq_dict:
                for j in range(seq[0].shape[1]):
                    
                    img_mask=seq[0][0][j]*255.0
                    img_mask=np.pad(img_mask,((0,0),(10,10)),'constant', constant_values=(0,0)) 

                    
                    label_path=os.path.join(path, label[0])
                    if not os.path.exists(label_path):
                        os.mkdir(label_path)
                    seq_type_path = os.path.join(label_path, seq_type[0])
                    if not os.path.exists(seq_type_path):
                        os.mkdir(seq_type_path)
                    view_path = os.path.join(seq_type_path, view[0])
                    if not os.path.exists(view_path):
                        os.mkdir(view_path)
                    out_path = os.path.join(view_path, label[0]+'-'+seq_type[0]+'-'+view[0]+'-'+'{:0>3d}.png'.format(j+1))
                    
                    print(out_path)
                    cv2.imwrite(out_path,img_mask)
            if seq_type not in gallery_seq_dict:
                
            
                for j in range(len(seq)):
                    #seq[j] = np.concatenate((seq[j],np.random.uniform(size=(1,1,64,44))),axis=1)
                    seq[j] = self.np2var(seq[j]).float()
                if batch_frame is not None:
                    batch_frame = self.np2var(batch_frame).int()
                # print(batch_frame, np.sum(batch_frame))
                
                feature, _ = self.encoder(*seq, batch_frame)
                
                '''transform = trans.Compose([
                        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    ])'''
                #print(*seq)
                #print(seq[0][0,0,0,:])
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                adversary = LinfMomentumIterativeAttack(
                            self.encoder, loss_fn=attack_cosine_distance(target=torch.ones(1).to(device)), eps=1.0,
                            nb_iter=20, eps_iter=0.05, decay_factor=1., clip_min=0.0, clip_max=1.0,
                            targeted=False)
                adv = adversary.perturb(*seq, batch_frame, feature)
            
                pad_top = 0
                pad_bottom = 0
                pad_left = 10
                pad_right = 10
                adv = F.pad(adv, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

                #adv = self.perturb(*seq, batch_frame, feature)
                for j in range(adv.shape[1]):
                    
                    img_mask=adv[0][j].cpu().numpy()*255.0
                    
                    label_path=os.path.join(path, label[0])
                    if not os.path.exists(label_path):
                        os.mkdir(label_path)
                    seq_type_path = os.path.join(label_path, seq_type[0])
                    if not os.path.exists(seq_type_path):
                        os.mkdir(seq_type_path)
                    view_path = os.path.join(seq_type_path, view[0])
                    if not os.path.exists(view_path):
                        os.mkdir(view_path)
                    out_path = os.path.join(view_path, label[0]+'-'+seq_type[0]+'-'+view[0]+'-'+'{:0>3d}.png'.format(j+1))
                    
                    print(out_path)
                    cv2.imwrite(out_path,img_mask)
                    

    def save(self):
        os.makedirs(osp.join('checkpoint', self.model_name), exist_ok=True)
        torch.save(self.encoder.state_dict(),
                   osp.join('checkpoint', self.model_name,
                            '{}-{:0>5}-encoder.ptm'.format(
                                self.save_name, self.restore_iter)))
        torch.save(self.optimizer.state_dict(),
                   osp.join('checkpoint', self.model_name,
                            '{}-{:0>5}-optimizer.ptm'.format(
                                self.save_name, self.restore_iter)))

    # restore_iter: iteration index of the checkpoint to load
    def load(self, restore_iter):
        self.encoder.load_state_dict(torch.load(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-encoder.ptm'.format(self.save_name, restore_iter))))
        self.optimizer.load_state_dict(torch.load(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-optimizer.ptm'.format(self.save_name, restore_iter))))
            
    '''def perturb(self, x, batch_frame, y):
        eps=2
        nb_iter=20
        eps_iter=0.1
        decay_factor=1.
        clip_min=0.0
        clip_max=1.0
        delta = torch.zeros_like(x)
        g = torch.zeros_like(x)

        delta = nn.Parameter(delta)

        for i in range(nb_iter):

            if delta.grad is not None:
                delta.grad.detach_()
                delta.grad.zero_()

            imgadv = x + delta
            outputs, _ = self.encoder(imgadv, batch_frame)
            loss_fn = attack_cosine_distance(target=torch.ones(1).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
            loss = loss_fn(outputs, y)
            loss.backward()

            g = decay_factor * g + normalize_by_pnorm(
                delta.grad.data, p=1)
          
            delta.data += eps_iter * torch.sign(g)
            delta.data = clamp(
                delta.data, min=-eps, max=eps)
            delta.data = clamp(
                x + delta.data, min=clip_min, max=clip_max) - x
            
        rval = x + delta.data
        return rval'''
   
    def generate(self, flag, batch_size=1):
        self.encoder.eval()
        source = self.test_source if flag == 'test' else self.train_source
        self.sample_type = 'all'
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        feature_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()
        path = "../output/"

        for i, x in enumerate(data_loader):
            seq, view, seq_type, label, batch_frame = x
            gallery_seq_dict = [['nm-01'], ['nm-02'], ['nm-03'], ['nm-04']]
            for j in range(seq[0].shape[1]):
                img_mask=seq[0][0][j]*255.0
                img_mask=np.pad(img_mask,((0,0),(10,10)),'constant', constant_values=(0,0)) 
                label_path=os.path.join(path, label[0])
                if not os.path.exists(label_path):
                    os.mkdir(label_path)
                seq_type_path = os.path.join(label_path, seq_type[0])
                if not os.path.exists(seq_type_path):
                    os.mkdir(seq_type_path)
                view_path = os.path.join(seq_type_path, view[0])
                if not os.path.exists(view_path):
                    os.mkdir(view_path)
                if seq_type not in gallery_seq_dict:
                    k = j + int(batch_frame / 40) + 1
                else:
                    k = j
                out_path = os.path.join(view_path, label[0]+'-'+seq_type[0]+'-'+view[0]+'-'+'{:0>3d}.png'.format(k+1))
                #out_path = os.path.join(view_path, label[0]+'-'+seq_type[0]+'-'+view[0]+'-'+'{:0>3d}.png'.format(j+1))
                
                print(out_path)
                cv2.imwrite(out_path,img_mask)
            if seq_type not in gallery_seq_dict:
                for j in range(len(seq)):
                    seq[j] = self.np2var(seq[j]).float()
                if batch_frame is not None:
                    batch_frame = self.np2var(batch_frame).int()  
                feature, _ = self.encoder(*seq, batch_frame)
                
                # Number of GPUs available. Use 0 for CPU mode.
                ngpu = 1
                
                # Decide which device we want to run on
                device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
                
                # Create the generator
                #netG = Generator(ngpu).to(device).eval()
                dim = 64
                netG = GoodGenerator(dim, dim*dim*3).to(device).eval()
                
                
                g_state_dict = torch.load("../../wgan-gp-pytorch/output/generator.pt")
                
                def remove_module_str_in_state_dict(state_dict):
                    state_dict_rename = OrderedDict()
                    for k, v in state_dict.items():
                        name = k.replace("module.", "") # remove `module.`
                        state_dict_rename[name] = v
                    return state_dict_rename
                netG.load_state_dict(remove_module_str_in_state_dict(g_state_dict))
                
                
                netD = GoodDiscriminator(dim).to(device).eval()
                d_state_dict = torch.load("../../wgan-gp-pytorch/output/discriminator.pt")
                netD.load_state_dict(remove_module_str_in_state_dict(d_state_dict))
                    
                #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                adversary = LinfMomentumIterativeAttack(
                            netG, netD, self.encoder, loss_fn=attack_cosine_distance(target=torch.ones(1).to(device)), eps=1.0,
                            nb_iter=100, eps_iter=0.1, decay_factor=1., clip_min=-2.0, clip_max=2.0,
                            targeted=False)
                adv = adversary.perturb(*seq, batch_frame, feature)
            
                pad_top = 0
                pad_bottom = 0
                pad_left = 10
                pad_right = 10
                adv = F.pad(adv[:,:,:,10:54], (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
                
                '''n = int(batch_frame/40) + 1
                #nz = 1000
                #noise = torch.randn(n, nz, 1, 1).cuda()
                noise = torch.randn(n, 128).cuda()
                adv = netG(noise)
                adv = adv.view(n, 3, dim, dim)
                adv = adv * 0.5 + 0.5'''
                #print(adv.shape)

                for j in range(adv.shape[0]):
                    img_mask=adv[j][0].detach().cpu().numpy()*255.0
                    label_path=os.path.join(path, label[0])
                    if not os.path.exists(label_path):
                        os.mkdir(label_path)
                    seq_type_path = os.path.join(label_path, seq_type[0])
                    if not os.path.exists(seq_type_path):
                        os.mkdir(seq_type_path)
                    view_path = os.path.join(seq_type_path, view[0])
                    if not os.path.exists(view_path):
                        os.mkdir(view_path)
                    out_path = os.path.join(view_path, label[0]+'-'+seq_type[0]+'-'+view[0]+'-'+'{:0>3d}.png'.format(j+1))
                    
                    print(out_path)
                    cv2.imwrite(out_path,img_mask)
                    
    
                    
    def generate_A(self, flag, batch_size=1):
        self.encoder.eval()
        source = self.test_source if flag == 'test' else self.train_source
        self.sample_type = 'all'
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        feature_list = list()
        view_list = list()
        label_list = list()
        path = "/GaitSet/output_a/"

        for i, x in enumerate(data_loader):
            seq, view, label, batch_frame = x
            gallery_seq_dict = [['00_4'],['45_4'],['90_4']]
            for j in range(seq[0].shape[1]):
                img_mask=seq[0][0][j]*255.0
                img_mask=np.pad(img_mask,((0,0),(10,10)),'constant', constant_values=(0,0)) 
                label_path=os.path.join(path, label[0])
                if not os.path.exists(label_path):
                    os.mkdir(label_path)
                view_path = os.path.join(label_path, view[0])
                if not os.path.exists(view_path):
                    os.mkdir(view_path)
                if view not in gallery_seq_dict:
                    k = j + int(batch_frame / 40) + 1
                else:
                    k = j
                out_path = os.path.join(view_path, label[0]+'-'+view[0]+'-'+'{:0>3d}.png'.format(k+1))
                #out_path = os.path.join(view_path, label[0]+'-'+seq_type[0]+'-'+view[0]+'-'+'{:0>3d}.png'.format(j+1))
                
                print(out_path)
                cv2.imwrite(out_path,img_mask)
            if view not in gallery_seq_dict:
                for j in range(len(seq)):
                    seq[j] = self.np2var(seq[j]).float()
                if batch_frame is not None:
                    batch_frame = self.np2var(batch_frame).int()  
                feature, _ = self.encoder(*seq, batch_frame)
                
                # Number of GPUs available. Use 0 for CPU mode.
                ngpu = 1
                
                # Decide which device we want to run on
                device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
                
                # Create the generator
                #netG = Generator(ngpu).to(device).eval()
                dim = 64
                netG = GoodGenerator(dim, dim*dim*3).to(device).eval()
                
                
                g_state_dict = torch.load("/GaitSet/model/generator.pt")
                
                def remove_module_str_in_state_dict(state_dict):
                    state_dict_rename = OrderedDict()
                    for k, v in state_dict.items():
                        name = k.replace("module.", "") # remove `module.`
                        state_dict_rename[name] = v
                    return state_dict_rename
                netG.load_state_dict(remove_module_str_in_state_dict(g_state_dict))
                
                
                netD = GoodDiscriminator(dim).to(device).eval()
                d_state_dict = torch.load("/GaitSet/model/discriminator.pt")
                netD.load_state_dict(remove_module_str_in_state_dict(d_state_dict))
                    
                #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                adversary = LinfMomentumIterativeAttack(
                            netG, netD, self.encoder, loss_fn=attack_cosine_distance(target=torch.ones(1).to(device)), eps=1.0,
                            nb_iter=100, eps_iter=0.1, decay_factor=1., clip_min=-2.0, clip_max=2.0,
                            targeted=False)
                adv = adversary.perturb(*seq, batch_frame, feature)
            
                pad_top = 0
                pad_bottom = 0
                pad_left = 10
                pad_right = 10
                adv = F.pad(adv[:,:,:,10:54], (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
                
                '''n = int(batch_frame/40) + 1
                #nz = 1000
                #noise = torch.randn(n, nz, 1, 1).cuda()
                noise = torch.randn(n, 128).cuda()
                adv = netG(noise)
                adv = adv.view(n, 3, dim, dim)
                adv = adv * 0.5 + 0.5'''
                #print(adv.shape)

                for j in range(adv.shape[0]):
                    img_mask=adv[j][0].detach().cpu().numpy()*255.0
                    label_path=os.path.join(path, label[0])
                    if not os.path.exists(label_path):
                        os.mkdir(label_path)
                    view_path = os.path.join(label_path, view[0])
                    if not os.path.exists(view_path):
                        os.mkdir(view_path)
                    out_path = os.path.join(view_path, label[0]+'-'+view[0]+'-'+'{:0>3d}.png'.format(j+1))
                    
                    print(out_path)
                    cv2.imwrite(out_path,img_mask)

    def evaluate(self, flag, batch_size=1):
        self.encoder.eval()
        source = self.test_source if flag == 'test' else self.train_source
        self.sample_type = 'all'
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        feature_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()
        

        for i, x in enumerate(data_loader):
            seq, view, seq_type, label, batch_frame = x
            gallery_seq_dict = [['nm-01'], ['nm-02'], ['nm-03'], ['nm-04']]
            if seq_type not in gallery_seq_dict:
                for j in range(len(seq)):
                    seq[j] = self.np2var(seq[j]).float()
                if batch_frame is not None:
                    batch_frame = self.np2var(batch_frame).int()  
                n = int(batch_frame / 40) + 1
                
                
                # Number of GPUs available. Use 0 for CPU mode.
                ngpu = 1
                
                # Decide which device we want to run on
                device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
                
                # Create the generator
                #netG = Generator(ngpu).to(device).eval()
                dim = 64
                netD = GoodDiscriminator(dim).to(device).eval()
                
                g_state_dict = torch.load("/GaitSet/model/discriminator.pt")
                
                def remove_module_str_in_state_dict(state_dict):
                    state_dict_rename = OrderedDict()
                    for k, v in state_dict.items():
                        name = k.replace("module.", "") # remove `module.`
                        state_dict_rename[name] = v
                    return state_dict_rename
                netD.load_state_dict(remove_module_str_in_state_dict(g_state_dict))
                
                pad_top = 0
                pad_bottom = 0
                pad_left = 10
                pad_right = 10
                

                for j in range(n+5):
                    #print(seq[0].shape)
                    image = seq[0][:,j:j+1,:,:]
                    #print(image.shape)
                    image = torch.cat((image, image, image), 1)
                    image = F.pad(image[:,:,:,:], (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
                    print(image.shape)
                    gen_cost = netD(image)
                    print(gen_cost)
                    