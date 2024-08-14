import numpy as np
import pandas as pd
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import HDBSCAN
from k_means_constrained import KMeansConstrained
import pickle
import math
import time
import matplotlib.pyplot as plt
from itertools import chain

import torchvision.transforms as transforms
from torchvision.datasets import MNIST

"""class ResidualBlock(torch.nn.Module):

    def __init__(self, channels):
        
        super(ResidualBlock, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=channels[0],
                                      out_channels=channels[1],
                                      kernel_size=(3, 3),
                                      stride=(2, 2),
                                      padding=1)
        self.conv_1_bn = torch.nn.BatchNorm2d(channels[1])
                                    
        self.conv_2 = torch.nn.Conv2d(in_channels=channels[1],
                                      out_channels=channels[2],
                                      kernel_size=(1, 1),
                                      stride=(1, 1),
                                      padding=0)   
        self.conv_2_bn = torch.nn.BatchNorm2d(channels[2])

        self.conv_shortcut_1 = torch.nn.Conv2d(in_channels=channels[0],
                                               out_channels=channels[2],
                                               kernel_size=(1, 1),
                                               stride=(2, 2),
                                               padding=0)   
        self.conv_shortcut_1_bn = torch.nn.BatchNorm2d(channels[2])

    def forward(self, x):
        shortcut = x
        
        out = self.conv_1(x)
        out = self.conv_1_bn(out)
        out = nn.ReLU()(out)

        out = self.conv_2(out)
        out = self.conv_2_bn(out)
        
        # match up dimensions using a linear function (no relu)
        shortcut = self.conv_shortcut_1(shortcut)
        shortcut = self.conv_shortcut_1_bn(shortcut)
        
        out += shortcut
        out = nn.ReLU()(out)

        return out

#self.track_layers = {'conv1': self.conv1, 'conv2': self.conv2, 'fc1': self.fc1, 'fc2': self.fc2}"""

"""class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 3)
        self.conv2 = nn.Conv2d(20, 10, 3)
        self.fc1 = nn.Linear(24*24*10, 32)
        self.fc2 = nn.Linear(32, classes)
        self.flatten = nn.Flatten()
        self.track_layers = {'conv1': self.conv1, 'conv2': self.conv2, 'fc1': self.fc1, 'fc2': self.fc2}

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        #x = nn.Softmax(dim=1)(x)
        return x"""

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding = 2)
        self.maxpool1 = nn.MaxPool2d(2,2) 
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.maxpool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, classes)
        self.flatten = nn.Flatten()
        self.track_layers = {'conv1': self.conv1, 'conv2': self.conv2, 'fc1': self.fc1, 'fc2': self.fc2}

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        #x = nn.Softmax(dim=1)(x)
        return x

    def get_track_layers():
        return self.track_layers

    def apply_parameters(self, parameters_dict):
        with torch.no_grad():
            for layer_name in parameters_dict:
                self.track_layers[layer_name].weight.data *= 0
                self.track_layers[layer_name].bias.data *= 0
                self.track_layers[layer_name].weight.data += parameters_dict[layer_name]['weight']
                self.track_layers[layer_name].bias.data += parameters_dict[layer_name]['bias']
        #print("yhn tk aara h")

    def get_parameters(self):
        parameters_dict = dict()
        for layer_name in self.track_layers:
            #print(layer_name)
            parameters_dict[layer_name] = {
                'weight': self.track_layers[layer_name].weight.data,
                'bias': self.track_layers[layer_name].bias.data
            }
        return parameters_dict

    def batch_accuracy(self, outputs, labels):
        with torch.no_grad():
            _, predictions = torch.max(outputs, dim=1)
            return torch.tensor(torch.sum(predictions == labels).item() / len(predictions))

    def _process_batch(self, batch):
        images, labels = batch
        outputs = self(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        accuracy = self.batch_accuracy(outputs, labels)
        return (loss, accuracy)

    def fit(self, dataset, epochs, lr, batch_size=128, opt=torch.optim.SGD):
        dataloader = DeviceDataLoader(DataLoader(dataset, batch_size, shuffle=True), device)
        optimizer = opt(self.parameters(), lr)
        history = []
        for epoch in range(epochs):
            losses = []
            accs = []
            for batch in dataloader:
                loss, acc = self._process_batch(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss.detach()
                losses.append(loss)
                accs.append(acc)
            avg_loss = torch.stack(losses).mean().item()
            avg_acc = torch.stack(accs).mean().item()
            history.append((avg_loss, avg_acc))
        return history

    def evaluate(self, dataset, batch_size=128):
        dataloader = DeviceDataLoader(DataLoader(dataset, batch_size), device)
        losses = []
        accs = []
        with torch.no_grad():
            for batch in dataloader:
                loss, acc = self._process_batch(batch)
                losses.append(loss)
                accs.append(acc)
        avg_loss = torch.stack(losses).mean().item()
        avg_acc = torch.stack(accs).mean().item()
        return (avg_loss, avg_acc)



class DeviceDataLoader(DataLoader):
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device

        def __iter__(self):
            for batch in self.dl:
                yield to_device(batch, self.device)

        def __len__(self):
            return len(self.dl)


class Client:
    def __init__(self, client_id, dataset):
        self.client_id = client_id
        self.dataset = dataset
        self.net = to_device(ConvNet(), device)

    def get_dataset_size(self):
        return len(self.dataset)

    def get_client_id(self):
        return self.client_id

    def get_weights(self):
        return self.net.get_parameters()

    def train(self, parameters_dict):
        self.net.apply_parameters(parameters_dict)
        train_history = self.net.fit(self.dataset, epochs_per_client, learning_rate, batch_size)
        print('{}: Loss = {}, Accuracy = {}'.format(self.client_id, round(train_history[-1][0], 4), round(train_history[-1][1], 4)))
        return self.net.get_parameters()




def get_dist(param1, param2):
    max_d = -1
    with torch.no_grad():
        pdist = nn.PairwiseDistance(p=2.0)
        for layer_name in param1:
            w_dist = torch.max(pdist(param1[layer_name]['weight'], param2[layer_name]['weight'])).item()
            b_dist = torch.max(pdist(param1[layer_name]['bias'], param2[layer_name]['bias'])).item()
            #print(w_dist.data[0])
            max_d = max(max_d, w_dist, b_dist)
    return max_d

#def cos(param1_list, param2_list):


def get_dist_cosine(param1, param2):
    cos = nn.CosineSimilarity(dim=0, eps=1e-8)
    i=0
    with torch.no_grad():
        for layer_name in param1:
            if i==0:
                param1_list = torch.flatten(param1[layer_name]['weight'])
                param1_list = torch.cat((param1_list, torch.flatten(param1[layer_name]['bias'])),0)
                param2_list = torch.flatten(param1[layer_name]['weight'])
                param2_list = torch.cat((param2_list, torch.flatten(param2[layer_name]['bias'])),0)
            else:
                param1_list = torch.cat((param1_list, torch.flatten(param1[layer_name]['weight'])),0)
                param1_list = torch.cat((param1_list, torch.flatten(param1[layer_name]['bias'])),0)
                param2_list = torch.cat((param2_list, torch.flatten(param2[layer_name]['weight'])),0)
                param2_list = torch.cat((param2_list, torch.flatten(param2[layer_name]['bias'])),0)
            i+=1
    return 1 - cos(param1_list, param2_list)

def get_dist_avg(param1, param2):
    dist=[]
    w_dist = []
    b_dist = []
    with torch.no_grad():
        pdist = nn.PairwiseDistance(p=2.0)
        for layer_name in param1:
            w_dist = w_dist + list(torch.flatten(pdist(param1[layer_name]['weight'], param2[layer_name]['weight'])))
            b_dist = b_dist + list(torch.flatten(pdist(param1[layer_name]['bias'], param2[layer_name]['bias'])))
    #dist = list(torch.flatten(w_dist))
    #dist.append(list(torch.flatten(b_dist)))
    dist = w_dist + b_dist
    return np.array(dist).mean()
# In[6]:

def apply_perturbation(param, delta_c):
        for layer_name in param:
            perturbation_weight = torch.normal(0, delta_c, size = param[layer_name]['weight'].size(), device='cuda')
            perturbation_bias = torch.normal(0, delta_c, size = param[layer_name]['bias'].size(), device='cuda')
            param[layer_name]['weight'] += perturbation_weight
            param[layer_name]['bias'] += perturbation_bias
        return param


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def fedavg_cluster(cluster, cluster_frac):
    avg_net = to_device(ConvNet(), device)
    new_parameters = dict([(layer_name, {'weight': 0, 'bias': 0}) for layer_name in avg_net.get_parameters()])
    for i in range(len(cluster)):
        client_parameters = cluster[i]
        #fraction = client.get_dataset_size() / total_train_size
        for layer_name in client_parameters:
            new_parameters[layer_name]['weight'] += cluster_frac[i] * client_parameters[layer_name]['weight']
            new_parameters[layer_name]['bias'] += cluster_frac[i] * client_parameters[layer_name]['bias']
    avg_net.apply_parameters(new_parameters)
    return avg_net.get_parameters()

train_data = MNIST(
    root = 'data',
    train = True,
    transform = transforms.ToTensor(),
    download = True,
)
test_data = MNIST(
    root = 'data',
    train = False,
    transform = transforms.ToTensor(),
    download = True
)


total_train_size = len(train_data)
total_test_size = len(test_data)
#total_dev_size = len(dev_data)

classes = 10
input_dim = 784

num_clients = 50
rounds = 50
batch_size = 128
epochs_per_client = 3
learning_rate = 2e-2

device = get_device()
# In[48]:


#below is for non-iid
device_file = open("device records.p", "rb")
random_num_size = pickle.load(device_file)
random_num_size[-1]+=(len(train_data)-sum(random_num_size))
print(random_num_size)

#examples_per_client = total_train_size // num_clients
client_datasets = random_split(train_data, list(random_num_size))
clients = [Client(i, client_datasets[i]) for i in range(num_clients)]
client_frac=[]
for client in clients:
    fraction = client.get_dataset_size() / total_train_size
    client_frac.append(client.get_dataset_size() / total_train_size)


ks = [4,6,8,10] #3,4,6,8,
xs = [2,3,4]
only_once = True

for k in ks:
    for x in xs:
        unlearning_time = []
        num_clients = 50
        clients = [Client(i, client_datasets[i]) for i in range(num_clients)]
        clients_id = list(range(num_clients))
        total_train_size = len(train_data)
        client_frac=[]
        for client in clients:
            client_frac.append(client.get_dataset_size() / total_train_size)
        if x>=k:
            break
        fedAvg_model = to_device(ConvNet(), device)
        history = []
        fedAvg_history = []
        train_acc_round = []
        test_acc_round = []
        fedavg_train_acc_round = []
        fedavg_test_acc_round = []
        distance_rounds = []
        for r in range(rounds):
            print('Start Round {} ... with k = {} x = {}'.format(r + 1, k, x))
            if r>1:
                 unlearn_prob = random.random()
                 if unlearn_prob >= 0.80:
                    
                    start = time.time()
                    unlearn_client_id = random.choice(clients_id)
                    #print("Unlearning begins for {}".format(unlearn_client_id))
                    unclient_ind = clients_id.index(unlearn_client_id)
                    total_train_size -= clients[unclient_ind].get_dataset_size()
                    del client_frac[unclient_ind]
                    del clients[unclient_ind]
                    clients_id.remove(unlearn_client_id)
                    #print("Number of clients left {}".format(len(clients)))
                    retrain = False
                    retrain_round = -1
                    for prev_round in range(r):
                        #print(prev_round)
                        prev_cluster_file = open("IPFedAvg_iid cluster indices at {} rounds with k = {} and x = {}.p".format(prev_round, k, x), "rb")
                        prev_cluster_ids = pickle.load(prev_cluster_file)
                        #print(prev_cluster_ids)
                        for i in range(len(prev_cluster_ids)):
                            if unlearn_client_id in prev_cluster_ids[i]:
                                #print("yhn tk aara h")
                                prev_cluster_ids[i].remove(unlearn_client_id)
                                pickle.dump(prev_cluster_ids, open("IPFedAvg_iid cluster indices at {} rounds with k = {} and x = {}.p".format(prev_round, k, x), "wb"))
                                if len(prev_cluster_ids[i])<x and retrain == False:
                                    #print("Yhn dikkat h {} {}".format(prev_round, x))
                                    retrain = True
                                    retrain_round = prev_round
                            """for j in range(len(prev_cluster_ids[i])):
                                                                                                                    if prev_cluster_ids[i][j]>unlearn_client_id:
                                                                                                                        prev_cluster_ids[i][j] -=1"""
                            if retrain:
                                break
                    if retrain:
                        print("retrain request came round = {} k = {} and x = {}".format(retrain_round, k, x))
                        for rU in range(retrain_round, r):
                            num_clients = len(clients)
                            dist_matrix = np.zeros([num_clients, num_clients])
                            unlearn_model_file = open("IPFedAvg_iid global model at {} rounds with k = {} and x = {}.p".format(retrain_round, k, x), "rb")
                            curr_parameters = pickle.load(unlearn_model_file)
                            i=0
                            for client in clients:
                                client_parameters = client.train(curr_parameters)
                                client_frac[i] = client.get_dataset_size() / total_train_size #updating client frac after unlearning
                                i=i+1

                            for i in range(len(dist_matrix)):
                                for j in range(i, len(dist_matrix)):
                                    temp = get_dist_cosine(clients[i].get_weights(), clients[j].get_weights())
                                    dist_matrix[i][j] = temp
                                    dist_matrix[j][i] = temp
                            dist_matrix = np.array(dist_matrix)
                            distance_rounds.append(dist_matrix.mean)
                            num_clusters = int(num_clients/k)
                            IP_clusters=[[] for i in range(num_clusters)]
                            cluster_clientIDs=[[] for i in range(num_clusters)]
                            IP_frac=[[] for i in range(num_clusters)]
                            clf = KMeansConstrained(
                            n_clusters=num_clusters,
                            size_min=k,
                            size_max=2*k,
                            random_state=0)
                            clf.fit(dist_matrix)
                            labels = np.array(clf.labels_)
                            medoid_maxDist = []
                            for i in range(num_clusters):
                                same_clus = np.where(labels == i)[0]
                                medoid_maxDist.append(np.max(dist_matrix[np.ix_(same_clus, same_clus)]))
                            for i in range(num_clients):
                                IP_clusters[labels[i]].append(clients[i].get_weights())
                                cluster_clientIDs[labels[i]].append(clients[i].get_client_id())
                                IP_frac[labels[i]].append(client_frac[i])
                            IP_models = []
                            gl_frac = []
                            for i in range(num_clusters):
                                ind = random.randint(0, len(IP_clusters[i])-1)

                                temp_model = apply_perturbation(IP_clusters[i][ind], medoid_maxDist[i])
                                IP_models.append(temp_model)
                                gl_frac.append(sum(IP_frac[i]))
                            fedAvg_model.apply_parameters(fedavg_cluster(IP_models, gl_frac))
                            pickle.dump(fedAvg_model.get_parameters(), open("IPFedAvg_iid global model at {} rounds with k = {}.p".format(rU, k, x), "wb"))
                            pickle.dump(cluster_clientIDs, open("IPFedAvg_iid cluster indices at {} rounds with k = {} and x = {}.p".format(rU, k, x), "wb"))
                    end = time.time()
                    unlearning_time.append(end-start)
            num_clients = len(clients)
            print("Number of clients left after unlearning {}".format(num_clients))
            dist_matrix = np.zeros([num_clients, num_clients])
            curr_parameters = fedAvg_model.get_parameters()
            i=0
            client_frac=[]
            for client in clients:
                client.train(curr_parameters)
                client_frac.append(client.get_dataset_size() / total_train_size) #updating client frac after unlearning
                i=i+1
            print("train hora h 1")
            for i in range(len(dist_matrix)):
                for j in range(i, len(dist_matrix)):
                    temp = get_dist_cosine(clients[i].get_weights(), clients[j].get_weights())
                    dist_matrix[i][j] = temp
                    dist_matrix[j][i] = temp
            dist_matrix = np.array(dist_matrix)
            distance_rounds.append(dist_matrix.mean)
            num_clusters = int(num_clients/k)
            IP_clusters=[[] for i in range(num_clusters)]
            cluster_clientIDs=[[] for i in range(num_clusters)]
            IP_frac=[[] for i in range(num_clusters)]
            print("train hora h 2")
            clf = KMeansConstrained(
            n_clusters=num_clusters,
            size_min=k, max_iter = 20,
            size_max=2*k,
            random_state=0)
            
            clf.fit(dist_matrix)
            
            labels = np.array(clf.labels_)
            
            print("train hora h 3")

            medoid_maxDist = []
            for i in range(num_clusters):
                same_clus = np.where(labels == i)[0]
                medoid_maxDist.append(np.max(dist_matrix[np.ix_(same_clus, same_clus)]))
            
            print(medoid_maxDist)
            print("train hora h 4")
            for i in range(num_clients):
                IP_clusters[labels[i]].append(clients[i].get_weights())
                cluster_clientIDs[labels[i]].append(clients[i].get_client_id())
                IP_frac[labels[i]].append(client_frac[i])
                #print(labels[i])
            IP_models = []
            gl_frac = []
            
            for i in range(num_clusters):
                ind = random.randint(0, len(IP_clusters[i])-1)
                temp_model = apply_perturbation(IP_clusters[i][ind], medoid_maxDist[i])
                IP_models.append(temp_model)
                gl_frac.append(sum(IP_frac[i]))

            
            
            fedAvg_model.apply_parameters(fedavg_cluster(IP_models, gl_frac))
            pickle.dump(fedAvg_model.get_parameters(), open("IPFedAvg_iid global model at {} rounds with k = {} and x = {}.p".format(r, k, x), "wb"))
            pickle.dump(cluster_clientIDs, open("IPFedAvg_iid cluster indices at {} rounds with k = {} and x = {}.p".format(r, k, x), "wb"))
            
            fedavg_train_loss, fedavg_train_acc = fedAvg_model.evaluate(train_data)
            
            fedavg_train_acc_round.append(fedavg_train_acc)
            
            fedavg_test_loss, fedavg_test_acc = fedAvg_model.evaluate(test_data)
            print("Fedavg Test Accuracy: {}\n".format(fedavg_test_acc))
            fedavg_test_acc_round.append(fedavg_test_acc)
            print('After round {}, train_loss = {}, train_acc= {}, test_acc = {}\n'.format(r + 1, round(fedavg_train_loss, 4),
                    round(fedavg_train_acc, 4), round(fedavg_test_acc, 4)))
            fedAvg_history.append(fedavg_train_loss)

        pickle.dump(fedavg_test_acc_round, open("FedAvg test accuracy avg after {} rounds with k = {} and x = {}.p".format(rounds, k, x), "wb"))
        pickle.dump(unlearning_time, open("FedAvg unlearning time after {} rounds with k = {} and x = {}.p".format(rounds, k, x), "wb"))