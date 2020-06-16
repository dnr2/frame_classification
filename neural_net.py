from __future__ import print_function
import math
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertForMaskedLM
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
        # used to compute probabilities
        self.softmax = nn.Softmax(dim=1)

    def fit(self, dataset, n_epochs = 1, batch_size = 10, verbose=True, print_every=20):
        # create data loader from data set
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True)

        # loop over the dataset multiple times
        for epoch in range(n_epochs):
            running_loss = 0.0
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
                
                running_loss += loss.item()
                if verbose:
                    # print statistics
                    # prints every "print_every" mini-batches      
                    if i % print_every == (print_every - 1):
                        print('[%2d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / print_every))
                        running_loss = 0.0
        
        print("Training finished")


    def test(self, dataset, batch_size=10):
        label_correct = defaultdict(lambda: 0)
        label_total = defaultdict(lambda: 0)

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for (inputs, labels) in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                predicted = self.predict(inputs)
                
                for (label, pred) in zip(labels.tolist(), predicted.tolist()):
                    label_correct[label] += 1 if label == pred else 0
                    label_total[label] += 1
        total = 0
        correct = 0
        for label in label_correct.keys():
            print("Label %s accuracy %d\n" %
                (label, 100 * label_correct[label] / label_total[label]))
            correct += label_correct[label]
            total += label_total[label]

        # TODO: only using accuracy, allow for other metrics
        print('Total accuracy on %d data points: %d %%' % (
            dataset.__len__(), 100 * correct / total))

    def predict_dataset(self, dataset, batch_size=10):
        predicted_lst = []
        probs_lst = []
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False)    
        with torch.no_grad():
            for (inputs, _) in data_loader:
                inputs = inputs.to(self.device)
                predicted, probs = self.predict(inputs)
                predicted_lst.append(predicted)
        predicted_tensor = torch.cat(predicted_lst, 0)
        probs_tensor = torch.cat(probs_lst, 0)
        return predicted_tensor, probs_tensor
    
    def predict(self, inputs):
        with torch.no_grad():
            outputs = self.net(inputs)
            probabilities = self.softmax(outputs)
            _, predicted = torch.max(outputs.data, 1)
            return predicted, probabilities[predicted]

    def predict_top_k_dataset(dataset, k, batch_size=1):
        predicted_lst = []
        probs_lst = []
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False)    
        with torch.no_grad():
            for (inputs, _) in data_loader:
                predicted, probs = predict_top_k(inputs, k)
                predicted_lst.append(predicted)
                probs_lst.append(probs)
        predicted_tensor = torch.cat(predicted_lst, 0)
        probs_tensor = torch.cat(probs_lst, 0)
        return predicted_tensor, probs_tensor
    
    def predict_top_k(inputs, k):
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = net(inputs)
            logits, predicted = torch.topk(outputs.data, k, dim = 1)
            probs = self.softmax(logits)
            return predicted, probs