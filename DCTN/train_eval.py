import os
import argparse
import tqdm
import numpy as np
from itertools import chain
from collections import OrderedDict
import torch
from sklearn.metrics import f1_score 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset
from utils import OfficeImage, LinePlotter
from model import Extractor, Classifier, Discriminator
from model import get_cls_loss, get_dis_loss, get_confusion_loss

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default="/home/xuruijia/ZJY/ADW/Office-31/")
parser.add_argument("-s1", default="amazon")
parser.add_argument("-s2", default="webcam")
parser.add_argument("-t", default="dslr")
parser.add_argument("--batch_size", default=32)
parser.add_argument("--shuffle", default=True)
parser.add_argument("--num_workers", default=8)
parser.add_argument("--steps", default=8)
parser.add_argument("--snapshot", default="/content/drive/MyDrive/MSDA/A_W_2_D_Open/bvlc_A_W_2_D/snapshot")
parser.add_argument("--s1_weight", default=0.5)
parser.add_argument("--s2_weight", default=0.5)
parser.add_argument("--lr", default=0.00001)
parser.add_argument("--beta1", default=0.9)
parser.add_argument("--beta2", default=0.999)
parser.add_argument("--gpu_id", default=0)
parser.add_argument("--num_classes", default=5)#########
parser.add_argument("--threshold", default=0.9)
parser.add_argument("--log_interval", default=5)
parser.add_argument("--cls_epoches", default=10)
parser.add_argument("--gan_epoches", default=5)
args = parser.parse_args()

data_root = args.data_root
batch_size = args.batch_size
shuffle = args.shuffle
num_workers = args.num_workers
steps = args.steps
snapshot = args.snapshot
s1_weight = args.s1_weight
s2_weight = args.s2_weight
lr = args.lr
beta1 = args.beta1
beta2 = args.beta2
gpu_id = args.gpu_id
num_classes = args.num_classes
threshold = args.threshold
log_interval = args.log_interval
cls_epoches = args.cls_epoches
gan_epoches = args.gan_epoches

import pandas as pd
import numpy as np
class RandomIntDataset(Dataset):
  def __init__(self, data, labels):
    # we randomly generate an array of ints that will act as data
    self.data = torch.tensor(data)
    # we randomly generate a vector of ints that act as labels
    self.labels = torch.tensor(labels)

  def __len__(self):
    # the size of the set is equal to the length of the vector
    return len(self.labels)

  def __str__(self):
    # we combine both data structures to present them in the form of a single table
    return str(torch.cat((self.data, self.labels.unsqueeze(1)), 1))

  def __getitem__(self, i):
  # the method returns a pair: given - label for the index number i
    return self.data[i], self.labels[i]

Source_train_1 = pd.read_csv("/content/drive/MyDrive/MSDA/A_W_2_D_Open/bvlc_A_W_2_D/data/Source_train_1.csv")
Source_test_1 = pd.read_csv("/content/drive/MyDrive/MSDA/A_W_2_D_Open/bvlc_A_W_2_D/data/Source_test_1.csv")

Source_train_2 = pd.read_csv("/content/drive/MyDrive/MSDA/A_W_2_D_Open/bvlc_A_W_2_D/data/Source_train_2.csv")
Source_test_2 = pd.read_csv("/content/drive/MyDrive/MSDA/A_W_2_D_Open/bvlc_A_W_2_D/data/Source_test_2.csv")

Target_train = pd.read_csv("/content/drive/MyDrive/MSDA/A_W_2_D_Open/bvlc_A_W_2_D/data/Target_train.csv")
Target_test = pd.read_csv("/content/drive/MyDrive/MSDA/A_W_2_D_Open/bvlc_A_W_2_D/data/Target_test.csv")
print('Before: ',len(Source_test_1), len(Source_train_1),len(Source_test_2), len(Source_train_2),  len(Target_test), len(Target_train))
Source_train_1 = Source_train_1.iloc[:5792,:]
Source_test_1 = Source_test_1.iloc[:192,:]
Source_train_2 = Source_train_2.iloc[:3776,:]
Source_test_2 = Source_test_2.iloc[:192,:]
Target_train = Target_train.iloc[:2848,:]
Target_test = Target_test.iloc[:128,:]
print('After: ', len(Source_test_1), len(Source_train_1), len(Source_test_2), len(Source_train_2), len(Target_test), len(Target_train))

Source_train_data = Source_train_1.drop(['labels'], axis= 1).values
Source_train_data = Source_train_data.astype(np.float32)
Source_train_labels = Source_train_1.labels.values.astype(np.float32)
train_source_1 = RandomIntDataset(Source_train_data, Source_train_labels)

Source_test_data = Source_test_1.drop(['labels'], axis= 1).values
Source_test_data = Source_test_data.astype(np.float32)
Source_test_labels = Source_test_1.labels.values.astype(np.float32)
test_source_1 = RandomIntDataset(Source_test_data, Source_test_labels)



Source_train_data = Source_train_2.drop(['labels'], axis= 1).values
Source_train_data = Source_train_data.astype(np.float32)
Source_train_labels = Source_train_2.labels.values.astype(np.float32)
train_source_2 = RandomIntDataset(Source_train_data, Source_train_labels)

Source_test_data = Source_test_2.drop(['labels'], axis= 1).values
Source_test_data = Source_test_data.astype(np.float32)
Source_test_labels = Source_test_2.labels.values.astype(np.float32)
test_source_2 = RandomIntDataset(Source_test_data, Source_test_labels)


Target_train_data = Target_train.drop(['labels'], axis= 1).values
Target_train_data = Target_train_data.astype(np.float32)
Target_train_labels = Target_train.labels.values.astype(np.float32)
train_target = RandomIntDataset(Target_train_data, Target_train_labels)

Target_test_data = Target_test.drop(['labels'], axis= 1).values
Target_test_data = Target_test_data.astype(np.float32)
Target_test_labels = Target_test.labels.values.astype(np.float32)
test_target = RandomIntDataset(Target_test_data, Target_test_labels)


s1_loader_raw = torch.utils.data.DataLoader(train_source_1, batch_size=batch_size,
    shuffle=shuffle, num_workers=num_workers)
s2_loader_raw = torch.utils.data.DataLoader(train_source_2, batch_size=batch_size,
    shuffle=shuffle, num_workers=num_workers)
t_loader_raw = torch.utils.data.DataLoader(train_target, batch_size=batch_size,
    shuffle=shuffle, num_workers=num_workers)
t_loader_test = torch.utils.data.DataLoader(test_target, batch_size=batch_size,
    shuffle=False, num_workers=num_workers)


extractor = Extractor().cuda(gpu_id)
extractor.load_state_dict(torch.load("/content/drive/MyDrive/MSDA/A_W_2_D_Open/bvlc_A_W_2_D/pretrain/model.pth")['state_dict'])
s1_classifier = Classifier(256,num_classes).cuda(gpu_id)
s2_classifier = Classifier(256,num_classes).cuda(gpu_id)
s1_classifier.load_state_dict(torch.load("/content/drive/MyDrive/MSDA/A_W_2_D_Open/bvlc_A_W_2_D/pretrain/model.pth")['state_dict_c'])
s2_classifier.load_state_dict(torch.load("/content/drive/MyDrive/MSDA/A_W_2_D_Open/bvlc_A_W_2_D/pretrain/model.pth")['state_dict_c'])
s1_t_discriminator = Discriminator().cuda(gpu_id)
s2_t_discriminator = Discriminator().cuda(gpu_id)


def print_log(step, epoch, epoches, lr, l1, l2, l3, l4, l5, l6, l7, l8, flag, ploter, count):
    print("Step [%d/%d] Epoch [%d/%d] lr: %f, s1_cls_loss: %.4f, s2_cls_loss: %.4f, s1_t_dis_loss: %.4f, " \
          "s2_t_dis_loss: %.4f, s1_t_confusion_loss_s1: %.4f, s1_t_confusion_loss_t: %.4f, " \
          "s2_t_confusion_loss_s2: %.4f, s2_t_confusion_loss_t: %.4f, selected_source: %s" \
          % (step, steps, epoch, epoches, lr, l1, l2, l3, l4, l5, l6, l7, l8, flag))
    '''ploter.plot("s1_cls_loss", "train", count, l1)
    ploter.plot("s2_cls_loss", "train", count, l2)
    ploter.plot("s1_t_dis_loss", "train", count, l3)
    ploter.plot("s2_t_dis_loss", "train", count, l4)
    ploter.plot("s1_t_confusion_loss_s1", "train", count, l5)
    ploter.plot("s1_t_confusion_loss_t", "train", count, l6)
    ploter.plot("s2_t_confusion_loss_s2", "train", count, l7)
    ploter.plot("s2_t_confusion_loss_t", "train", count, l8)'''


count = 0
max_correct = 0
max_step = 0
max_epoch = 0
ploter = LinePlotter(env_name="bvlc_A_W_2_D")
for step in range(steps):
    # Part 1: assign psudo-labels to t-domain and update the label-dataset
    print("#################### Part1 ####################")
    extractor.eval()
    s1_classifier.eval()
    s2_classifier.eval()
    
    #fin = open(t_label)
    #fout = open(os.path.join(data_root, args.t, "pseudo/pse_label_" + str(step) + ".txt"), "w")
    if step > 0:
        s1_weight = s1_weight_loss / (s1_weight_loss + s2_weight_loss)
        s2_weight = s2_weight_loss / (s1_weight_loss + s2_weight_loss)
    print("s1_weight is: ", s1_weight)
    print("s2_weight is: ", s2_weight)
    list_label = []
    for i, (t_imgs, t_labels) in tqdm.tqdm(enumerate(t_loader_test)):
        t_imgs = Variable(t_imgs.cuda(gpu_id))
        t_feature = extractor(t_imgs)
        s1_cls = s1_classifier(t_feature)
        s2_cls = s2_classifier(t_feature)
        s1_cls = F.softmax(s1_cls)
        s2_cls = F.softmax(s2_cls)
        s1_cls = s1_cls.data.cpu().numpy()
        s2_cls = s2_cls.data.cpu().numpy()
        try:
          t_pred = s1_cls * s1_weight + s2_cls * s2_weight
        except:
          t_pred = s1_cls * s1_weight.item() + s2_cls * s2_weight.item()

        ids = t_pred.argmax(axis=1)
        list_label+= list(ids)
        '''for j in range(ids.shape[0]):
            #line = fin.next()
            #data = line.strip().split(" ")
            if t_pred[j, ids[j]] >= threshold:
                fout.write(data[0] + " " + str(ids[j]) + "\n")'''

    list_label = pd.DataFrame(list_label)
    #fin.close()
    #fout.close()     

  
    # Part 2: train F1t, F2t with pseudo labels
    print("#################### Part2 ####################")
    extractor.train()
    s1_classifier.train()
    s2_classifier.train()
    #t_pse_label = os.path.join(data_root, args.t, "pseudo/pse_label_" + str(step) + ".txt")
    #t_pse_set = OfficeImage(t_root, t_pse_label, split="train")

    Target_train_data = Target_train.drop(['labels'], axis= 1).values
    Target_train_data = Target_train_data.astype(np.float32)
    Target_train_labels = list_label.values.astype(np.float32)
    #print(Target_train_labels[:,0].shape)
    train_target = RandomIntDataset(Target_train_data, Target_train_labels[:,0])
    t_pse_loader_raw = torch.utils.data.DataLoader(train_target, batch_size=batch_size,
                           shuffle=shuffle, num_workers=num_workers)
    #print("Length of pseudo-label dataset: ", len(train_target))

    optim_extract = optim.Adam(extractor.parameters(), lr=lr, betas=(beta1, beta2))
    optim_s1_cls = optim.Adam(s1_classifier.parameters(), lr=lr, betas=(beta1, beta2))
    optim_s2_cls = optim.Adam(s2_classifier.parameters(), lr=lr, betas=(beta1, beta2))

    for cls_epoch in range(cls_epoches):
        s1_loader, s2_loader, t_pse_loader = iter(s1_loader_raw), iter(s2_loader_raw), iter(t_pse_loader_raw)
        for i, (t_pse_imgs, t_pse_labels) in tqdm.tqdm(enumerate(t_pse_loader)):
            try:
                s1_imgs, s1_labels = next(s1_loader) 
            except StopIteration:
                s1_loader = iter(s1_loader_raw)
                s1_imgs, s1_labels = next(s1_loader)
            try:
                s2_imgs, s2_labels = next(s2_loader)
            except StopIteration:
                s2_loader = iter(s2_loader_raw)
                s2_imgs, s2_labels = next(s2_loader)
            s1_imgs, s1_labels = Variable(s1_imgs.cuda(gpu_id), requires_grad=True), Variable(s1_labels.cuda(gpu_id), requires_grad=True)
            s2_imgs, s2_labels = Variable(s2_imgs.cuda(gpu_id), requires_grad=True), Variable(s2_labels.cuda(gpu_id), requires_grad=True)
            t_pse_imgs, t_pse_labels = Variable(t_pse_imgs.cuda(gpu_id), requires_grad=True), Variable(t_pse_labels.cuda(gpu_id), requires_grad=True)
            
            s1_t_imgs = torch.cat((s1_imgs, t_pse_imgs), 0)
            #print(t_pse_labels.size(), s1_labels.size())
            s1_t_labels = torch.cat((s1_labels, t_pse_labels), 0)
            s2_t_imgs = torch.cat((s2_imgs, t_pse_imgs), 0)
            s2_t_labels = torch.cat((s2_labels, t_pse_labels), 0)

            optim_extract.zero_grad()
            optim_s1_cls.zero_grad()
            optim_s2_cls.zero_grad()

            s1_t_feature = extractor(s1_t_imgs)
            s2_t_feature = extractor(s2_t_imgs)
            s1_t_cls = s1_classifier(s1_t_feature)
            s2_t_cls = s2_classifier(s2_t_feature)

            s2_t_cls = s2_t_cls.argmax(axis=1)######
            s1_t_cls = s1_t_cls.argmax(axis=1)######
            #print(s2_t_cls)
            s1_t_cls_loss = get_cls_loss(s1_t_labels, s1_t_cls)
            s2_t_cls_loss = get_cls_loss(s2_t_labels, s2_t_cls)

            torch.autograd.backward([s1_t_cls_loss, s2_t_cls_loss])

            optim_s1_cls.step()
            optim_s2_cls.step()
            optim_extract.step()

            if (i+1) % log_interval == 0:
                #print_log(step+1, cls_epoch+1, cls_epoches, lr, s1_t_cls_loss.data, \
                           #s2_t_cls_loss.data, 0, 0, 0, 0, 0, 0, "...", ploter, count)
                count += 1
    
        extractor.eval()
        s1_classifier.eval()
        s2_classifier.eval()
        correct = 0
        fcorrect = 0
        for (imgs, labels) in t_loader_test:
            imgs = Variable(imgs.cuda(gpu_id))
            imgs_feature = extractor(imgs)
          
            s1_cls = s1_classifier(imgs_feature)
            s2_cls = s2_classifier(imgs_feature)
            s1_cls = F.softmax(s1_cls)
            s2_cls = F.softmax(s2_cls)
            s1_cls = s1_cls.data.cpu().numpy()
            s2_cls = s2_cls.data.cpu().numpy()

            try:
              res = s1_cls * s1_weight + s2_cls * s2_weight
            except:
              res = s1_cls * s1_weight.item() + s2_cls * s2_weight.item()
            
            
            
            pred = res.argmax(axis=1)
            labels = labels.numpy()
            correct += np.equal(labels, pred).sum()
            fcorrect +=f1_score(labels, pred, average='weighted')
        current_fscore = fcorrect*args.batch_size/len(test_target)
        current_accuracy = correct * 1.0 / len(test_target)
        print("Current accuracy is: ", current_accuracy)
        print("Current F1 score is: ", current_fscore)
        if current_accuracy >= max_correct:
            max_correct = current_accuracy
            max_step = step
            max_epoch = cls_epoch
            torch.save(extractor.state_dict(), os.path.join(snapshot, "p2_extractor_" + str(step) + "_" + str(cls_epoch) + ".pth"))
            torch.save(s1_classifier.state_dict(), os.path.join(snapshot, "p2_s1_cls_" + str(step) + "_" + str(cls_epoch) + ".pth"))
            torch.save(s2_classifier.state_dict(), os.path.join(snapshot, "p2_s2_cls_" + str(step) + "_" + str(cls_epoch) + ".pth"))
            
         
    # Part 3: train discriminator and generate mix feature
    print("#################### Part3 ####################")
    extractor.train()
    s1_classifier.train()
    s2_classifier.train()
    optim_extract = optim.Adam(extractor.parameters(), lr=lr, betas=(beta1, beta2))
    optim_s1_t_dis = optim.Adam(s1_t_discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    optim_s2_t_dis = optim.Adam(s2_t_discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    s1_weight_loss = 0
    s2_weight_loss = 0
    for gan_epoch in range(gan_epoches):
        s1_loader, s2_loader, t_loader = iter(s1_loader_raw), iter(s2_loader_raw), iter(t_loader_raw)
        for i, (t_imgs, t_labels) in tqdm.tqdm(enumerate(t_loader)):
            
            s1_imgs, s1_labels = next(s1_loader)
            try:
              s2_imgs, s2_labels = next(s2_loader)
              #print(next(s2_loader))
            except StopIteration:
              pass
            s1_imgs, s1_labels = Variable(s1_imgs.cuda(gpu_id), requires_grad=True), Variable(s1_labels.cuda(gpu_id), requires_grad=True)
            s2_imgs, s2_labels = Variable(s2_imgs.cuda(gpu_id), requires_grad=True), Variable(s2_labels.cuda(gpu_id), requires_grad=True)
            t_imgs = Variable(t_imgs.cuda(gpu_id), requires_grad=True)
  
            extractor.zero_grad()
            s1_feature = extractor(s1_imgs)
            s2_feature = extractor(s2_imgs)
            t_feature = extractor(t_imgs)
            s1_cls = s1_classifier(s1_feature)
            s2_cls = s2_classifier(s2_feature)
            s1_t_fake = s1_t_discriminator(s1_feature)
            s1_t_real = s1_t_discriminator(t_feature)
            s2_t_fake = s2_t_discriminator(s2_feature)
            s2_t_real = s2_t_discriminator(t_feature)
            #print(s1_cls)
            #print(s2_t_cls)

            s1_cls = s1_cls.argmax(axis=1)######
            s2_cls = s2_cls.argmax(axis=1)######

            s1_cls_loss = get_cls_loss(s1_labels, s1_cls)
            s2_cls_loss = get_cls_loss(s2_labels, s2_cls)
            s1_t_dis_loss = get_dis_loss(s1_t_fake, s1_t_real)
            s2_t_dis_loss = get_dis_loss(s2_t_fake, s2_t_real)
            s1_weight_loss += s1_t_dis_loss.data
            s2_weight_loss += s2_t_dis_loss.data

            s1_t_confusion_loss_s1 = get_confusion_loss(s1_t_fake)
            s1_t_confusion_loss_t = get_confusion_loss(s1_t_real)            
            s1_t_confusion_loss = 0.5 * s1_t_confusion_loss_s1 + 0.5 * s1_t_confusion_loss_t

            s2_t_confusion_loss_s2 = get_confusion_loss(s2_t_fake)
            s2_t_confusion_loss_t = get_confusion_loss(s2_t_real)
            s2_t_confusion_loss = 0.5 * s2_t_confusion_loss_s2 + 0.5 * s2_t_confusion_loss_t

            if s1_t_dis_loss.data > s2_t_dis_loss.data:
                SELECTIVE_SOURCE = "S1"
                torch.autograd.backward([s1_cls_loss, s2_cls_loss, s1_t_confusion_loss])
            else:
                SELECTIVE_SOURCE = "S2"
                torch.autograd.backward([s1_cls_loss, s2_cls_loss, s2_t_confusion_loss])
            optim_extract.step()

            s1_t_discriminator.zero_grad()
            s2_t_discriminator.zero_grad()
            s1_t_fake = s1_t_discriminator(s1_feature.detach())
            s1_t_real = s1_t_discriminator(t_feature.detach())
            s2_t_fake = s2_t_discriminator(s2_feature.detach())
            s2_t_real = s2_t_discriminator(t_feature.detach())
            s1_t_dis_loss = get_dis_loss(s1_t_fake, s1_t_real)
            s2_t_dis_loss = get_dis_loss(s2_t_fake, s2_t_real)
            torch.autograd.backward([s1_t_dis_loss, s2_t_dis_loss])
            optim_s1_t_dis.step()
            optim_s2_t_dis.step()

            if (i+1) % log_interval == 0:
                #print_log(step+1, gan_epoch+1, gan_epoches, lr, s1_cls_loss.data, s2_cls_loss.data, s1_t_dis_loss.data, \
                          #s2_t_dis_loss.data, s1_t_confusion_loss_s1.data, s1_t_confusion_loss_t.data, \
                          #s2_t_confusion_loss_s2.data, s2_t_confusion_loss_t.data, SELECTIVE_SOURCE, ploter, count)
                count += 1

print("max_correct is :",str(max_correct))
print("max_step is :",str(max_step+1))
print("max_epoch is :",str(max_epoch+1))
#ploter.save()
