import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
# import torchvision.transforms as transforms
from torchvision import transforms as T
import random
from math import *
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter 
import csv


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=8, stride=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 8, 4)
        # self.conv3 = nn.Conv2d(8, 16, 4)
        self.fc1 = nn.Linear(512, 32)
        self.fc2 = nn.Linear(32, 2)
        

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.dropout(x, p=0.5, training=True)
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))        
        x = self.fc2(x)
        return x

class MyDataset(Dataset):
    def __init__(self, txt_path, transform = None, target_transform = None):
        
        with open(txt_path, 'r') as f:
            fh = f.readlines()

        labels = ['Male', 'Smiling']
                    
                     
        temp = fh[1].rstrip().split()
        
        label_index = [i + 1 for i in range(len(temp)) if temp[i] in labels]

        fh = fh[2:50002]       
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            tmp = [words[keep_label] for keep_label in label_index]
            tmp = [int(i) if int(i) > 0 else int(0) for i in tmp]
            imgs.append([words[0], tmp])

        self.imgs = imgs 
        self.transform = transform
        self.target_transform = target_transform

    
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        fn = os.path.join('/home/jindi/stargan-master/data/celeba/images/', fn)
        img = Image.open(fn)
        # img = np.asarray(img)

        if self.transform is not None:
            img = self.transform(img) 
        return [img, label]    

    def __len__(self):
	    return len(self.imgs)


def train(dataset, train_index_path):
    
    print('preparing training set ... ')
    train_index = np.load('train_index.npy')
    trainset = []
    for index in train_index:
        trainset.append(dataset.__getitem__(index))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    
    print('preparing net ... ')
    net = Net()
    net.to(device)

    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    # criterion = nn.BCELoss() 
    # F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    # optimizer = optim.SGD(net.parameters(), lr=0.0001)

    PATH_to_log_dir = './logs/'
    if not os.path.exists(PATH_to_log_dir):
                os.makedirs(PATH_to_log_dir)
    writer = SummaryWriter(PATH_to_log_dir)

    print('strat training ... ')

    for epoch in range(10000):  # loop over the dataset multiple times
        running_loss = 0.0
        print('-------------- epoch : %s -------------- ' % str(epoch))    
        i = epoch * len(trainloader) 
        # for i, data in enumerate(trainloader):
        # for data in tqdm(trainloader, ncols=100):
        for data in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), torch.stack(data[1], dim=1).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels.float())
            loss.float()
            loss.backward()
            optimizer.step()

            writer.add_scalar('Train/Loss', loss.item(), i)
            writer.flush()

            # print statistics
            running_loss = loss.item()
            if i % 200 == 199:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.5f' %
                    (epoch + 1, i + 1, running_loss))
                running_loss = 0.0

            i = i + 1

        if (epoch + 1) % 50 == 0:
            if not os.path.exists('./ckp'):
                os.makedirs('./ckp')
            PATH = './ckp/imgclf_checkpoint%s.pth' % str(epoch + 1)
            torch.save(net.state_dict(), PATH)


    print('Finished Training')

def forward_sample(model, criterion, input, targets):
    output = model(input)
    sub_loss = criterion(output, target)

    return output, sub_loss


def calc_accuracy(outputs, targets, score_thres, top_k=(1,)):
    max_k = max(top_k)
    accuracy = []
    thres_list = eval(score_thres)
    if isinstance(thres_list, float) or isinstance(thres_list, int) :
        thres_list = [eval(score_thres)]*len(targets)

    for i in range(len(targets)):
        target = targets[i]
        output = outputs[i].data
        batch_size = output.size(0)
        curr_k = min(max_k, output.size(1))
        top_value, index = output.cpu().topk(curr_k, 1)
        index = index.t()
        top_value = top_value.t()
        correct = index.eq(target.cpu().view(1,-1).expand_as(index))
        mask = (top_value>=thres_list[i])
        correct = correct * mask
        #print "masked correct: ", correct
        res = defaultdict(dict)
        for k in top_k:
            k = min(k, output.size(1))
            correct_k = correct[:k].view(-1).float().sum(0)[0]
            res[k]["s"] = batch_size
            res[k]["r"] = correct_k
            res[k]["ratio"] = float(correct_k)/batch_size
        accuracy.append(res)
    return accuracy


def test(dataset, test_index_path):
    with open('results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)    
        sum_batch = 0 
        accuracy = list()
        avg_loss = list()

        labelslist = ['Male', 'Smiling']

        writer.writerow([i for i in labelslist])

        print('preparing test set ... ')
        test_index = np.load('test_index.npy')
        testset = []
        for index in test_index:
            testset.append(dataset.__getitem__(index))

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        testloader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=200)
        
        print('preparing net ... ')
        net = Net()
        # net.to(device)
        criterion = nn.BCEWithLogitsLoss()

        Path = './ckp'
        for ind in range(1, len(os.listdir(Path))):
            ind = ind * 50
            ckp = "imgclf_checkpoint%s.pth" % str(ind)
            print('----------------------- %s --------------------------' % str(ckp))
            net.load_state_dict(torch.load(os.path.join(Path, ckp)))
            dataiter = iter(testloader)
            outputs = []
            labels = []
            
            
            for data in tqdm(testloader, ncols=100): 
                # images, label = data[0].to(device), torch.stack(data[1], dim=1).to(device)
                images, label = data[0], torch.stack(data[1], dim=1)
                output = net(images)
                y = torch.zeros(1, 2)
                x = torch.ones(1, 2)
                output = torch.where(output > 0, x, y)  # 1 - output > 0.5, x, y 看看这里是不是都是正数
                outputs.append(output)
                labels.append(label)
            outputs = torch.cat(outputs, dim=1)
            labels = torch.cat(labels, dim=1)
            outputs = outputs.view(-1, 8)
            labels = labels.view(-1, 8)
            outarr = outputs.detach().numpy()
            labelarr = labels.detach().numpy()

            result = outarr == labelarr
            num = result.shape[0]
            accs = []
            for i in range(len(labelslist)):
                tmp = result[:, i]
                mask = (tmp == True)
                tmp = tmp[mask]
                acc = tmp.size / num
                print('ACC of %s is %f' % (labelslist[i], acc)) 
                accs.append(acc)

            writer.writerow([float(i) for i in accs])

  


if __name__ == "__main__":
    
    datatxt_path = '/home/jindi/stargan-master/data/celeba/list_attr_celeba.txt'
    transform = []
    transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(178))
    transform.append(T.Resize(128))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    dataset = MyDataset(datatxt_path, transform=transform)

    train_index_path = 'train_index.npy'
    test_index_path = 'test_index.npy'
    
    # indexlist = np.arange(dataset.__len__())
    # np.random.shuffle(indexlist)
    # train_index = indexlist[:floor(dataset.__len__() * 0.8)]
    # test_index = np.setdiff1d(indexlist,train_index)

    # train_index = np.array(train_index)
    # test_index = np.array(test_index)

    # np.save('train_index.npy', train_index)
    # np.save('test_index.npy', test_index)
 

    # train(dataset, train_index_path)
                            
    test(dataset, test_index_path)

    # Path = './ckp'
    # for ckp in os.listdir(Path):
    #     net.load_state_dict(torch.load(os.path.join(Path, ckp)))
    #     outputs = net(images)
        

   
