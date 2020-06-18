from __future__ import print_function
import math
import numpy as np
import json
import time
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import defaultdict

# You should build your custom dataset as below.
class NpClassDataset(torch.utils.data.Dataset):

    def __init__(self, inputs, labels):
        """
        arguments should be numpy arrays with shapes:
        inputs: (N, F)
        labels: (N, 1)
        Where N = number of data points and F = number of features
        """
        self.inputs = inputs
        self.labels = labels
        
    def __getitem__(self, index):
        x = torch.from_numpy(self.inputs[index].astype(np.float32))
        y = torch.from_numpy(np.squeeze(self.labels[index]).astype(np.longlong))
        return x, y
        
    def __len__(self):        
        return self.labels.shape[0]

class Model():

    def __init__(self, net, criterion, optimizer):
        # try to move model to GPU, if exists
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')

        self.net = net
        self.net.to(self.device)
        self.criterion = criterion 
        self.optimizer = optimizer
        self.softmax = nn.Softmax(dim=1)

    def fit(self, dataset, n_epochs = 1, batch_size = 32, verbose=True, print_every=20):
        # create data loader from data set
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True)

        # loop over the dataset multiple times
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            running_loss = 0.0
            epoch_start_time = time.time()
            for i, (inputs, labels) in enumerate(data_loader, 0):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                running_loss += loss.item()
                if verbose:
                    # print statistics
                    # prints every "print_every" mini-batches      
                    if i % print_every == (print_every - 1):
                        print('[%2d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / print_every))
                        running_loss = 0.0
            
            print('Epoch %2d finished. Loss: %.3f (elapsed %.3fs)' % (
                epoch + 1, epoch_loss / len(data_loader), time.time() - epoch_start_time))

        print("Training finished")


    def test(self, dataset, batch_size=1):
        correct = 0
        total = 0
        
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False)
        
        label_accuracy = {}

        with torch.no_grad():
            for (inputs, labels) in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                # TODO: only using accuracy, allow for other metrics
                correct += (predicted == labels).sum().item()

        print('Accuracy on the %d data points: %d %%' % (
            dataset.__len__(), 100 * correct / total))


    def predict_dataset(self, dataset, batch_size=1):
        predicted_lst = []
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False)    
        with torch.no_grad():
            for (inputs, _) in data_loader:
                inputs = inputs.to(self.device)
                predicted = self.predict(inputs)
                predicted_lst.append(predicted)
        predicted_tensor = torch.cat(predicted_lst, 0)
        return predicted_tensor
    
    def predict(self, inputs):
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            return predicted

    def predict_top_k_dataset(self, dataset, k, batch_size=1):
        predicted_lst = []
        probs_lst = []
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False)    
        with torch.no_grad():
            for (inputs, _) in data_loader:
                inputs = inputs.to(self.device)
                predicted, probs = self.predict_top_k(inputs, k)
                predicted_lst.append(predicted)
                probs_lst.append(probs)
        predicted_tensor = torch.cat(predicted_lst, 0)
        probs_tensor = torch.cat(probs_lst, 0)
        return predicted_tensor, probs_tensor
    
    def predict_top_k(self, inputs, k):
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.net(inputs)
            logits, predicted = torch.topk(outputs.data, k, dim = 1)
            probs = self.softmax(logits)
            return predicted, probs

    def get_probabilities_dataset(self, dataset, filter_lst=None, batch_size=1):        
        """
        Gets the probability distribution for output classes for each point in 
        dataset.
        """        
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False)        
        batch_idx = 0
        probs_lst = []
        with torch.no_grad():
            for (inputs, _) in data_loader:
                inputs = inputs.to(self.device)
                # break down filter list according to running batch
                filter_lst_batch = filter_lst[batch_idx : batch_idx + batch_size]
                probs = self.get_probabilities(inputs, filter_lst_batch)
                probs_lst.append(probs)
                batch_idx += batch_size
        probs_tensor = torch.cat(probs_lst, 0)
        return probs_tensor

    def get_probabilities(self, inputs, filter_lst=None):
        """
        Gets the probability distribution for output classes.        
        """
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.net(inputs)            
            if filter_lst:
                # TODO: assumes inputs are 2-order tensors
                # for each outputs[i], zeros out elements with indexes not 
                # in filter[i] list.
                filter_tensor = torch.ones(*outputs.size()).to(self.device)
                filter_tensor = filter_tensor * 10.0
                for i, f in enumerate(filter_lst):
                    filter_tensor[i][f] = 1.0
                outputs = outputs * filter_tensor

            probs = self.softmax(outputs)
            return probs

class FNClassModel(Model):

    def __init__(self, dataset, model_weights_filepath):
        # Run training & testing
        net = self.create_net(input_dim = dataset.INPUT_DIM, output_dim = dataset.OUTPUT_DIM)
        criterion = nn.CrossEntropyLoss(),
        optimizer = optim.Adam(net.parameters(), lr=10e-5)
        super().__init__(net, criterion, optimizer)
        self.load_net_from_file(model_weights_filepath)

    def create_net(self, input_dim, output_dim):
        layers = [
            nn.Dropout(),
            nn.Linear(input_dim, 400),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(400, output_dim),
        ]
        net = nn.Sequential(*layers)
        return net

    def load_net_from_file(self, filepath):
        self.net.load_state_dict(torch.load(filepath))
