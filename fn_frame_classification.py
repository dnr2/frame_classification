from __future__ import print_function
import os
import math
import numpy as np
import json
import time
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from make_fn_data import load_fn_data
from neural_net import Model, NpClassDataset, FNClassModel
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from settings import FRAME_DICT, FN_CLASSIFIER_WEIGHTS_FILE

# Custom pytorch dataset for handing Frame Net classification with BERT.
class FnBertDataset(torch.utils.data.Dataset):

    def __init__(self, inputs = [], labels = [], tokenizer = None, bert_model = None):
        """
        Arguments format:
        inputs: [(text1, start1, end1), ...]
        labels: [label_id1, ...]        
        """
        self.inputs = inputs
        self.labels = labels
        
        # use best available resource to run BERT model.
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')

        # Setup BERT model and Tokenizer.
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if bert_model:
            self.bert_model = bert_model
        else:
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')        
        self.bert_model.eval()
        self.bert_model.to(self.device)

        # Dataset constants
        self.MAX_LEN = 4
        self.INPUT_DIM = self.MAX_LEN * self.bert_model.config.hidden_size
        self.OUTPUT_DIM = len(FRAME_DICT.keys())

    def __getitem__(self, index):
        text, start, end = self.inputs[index]
        x = self.get_bert_hidden_state(text, start, end)
        y = torch.tensor(self.labels[index]).long()        
        return x, y
        
    def __len__(self):
        return len(self.labels)
    
    def set_input_only(self, inputs):
        self.inputs = inputs
        self.labels = [0] * len(inputs)

    def get_bert_hidden_state(self, text, start, end):
        text = "[CLS] " + text + " [SEP]"
        start += len("[CLS] ")
        end += len("[CLS] ")
        
        # Compute start end end using token indexes
        tk_start, tk_end = self.pos_to_token_idx(text, start, end)
        tk_end = min(tk_start + self.MAX_LEN, tk_end)
        # Tokenize input
        tokenized_text = self.tokenizer.tokenize(text)
    
        # Convert token to vocabulary indices
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
        # Predict hidden states features for each layer
        with torch.no_grad():
            outputs = self.bert_model(tokens_tensor)
            # Hidden state of the last layer of the Bert model
            hidden = torch.squeeze(outputs[0], dim = 0)
            # Slice hidden state to hidden[start:end]
            hidden = hidden.narrow(0, tk_start, tk_end-tk_start)
            # Add padding
            pad = torch.zeros(self.MAX_LEN, hidden.size()[1])            
            pad[0:hidden.size()[0],:] = hidden
            hidden = torch.flatten(pad)
            return hidden

    def pos_to_token_idx(self, text, start, end):
        target_prefix = self.tokenizer.tokenize(text[:start])
        target = self.tokenizer.tokenize(text[start:end+1])
        tk_start = len(target_prefix)
        tk_end = tk_start + len(target)
        return tk_start, tk_end

class FrameNetClassifier():

    def __init__(self):
        self.dataset = FnBertDataset()
        model_weights_filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), FN_CLASSIFIER_WEIGHTS_FILE)
        self.model = FNClassModel(self.dataset, model_weights_filepath)
        self.model.net.eval()
        self.frame_dict_rev = {v: k for k, v in FRAME_DICT.items()}

    def predict_top_k_frames(self, input_data, k = 5):
        """
        input_data: input with format [(sentence_1, start_1, end_1), ...]
        """
        self.dataset.set_input_only(input_data)
        preds, probs = self.model.predict_top_k_dataset(self.dataset, k)
        preds = preds.tolist()
        probs = probs.tolist()
        # map predictions to frame names
        preds = [[self.frame_dict_rev[idx] for idx in pred] for pred in preds]
        for inputs, pred, prob in zip(input_data, preds, probs):
            sentence, start, end = inputs
            print("Top frames for: '%s' in '%s'" % (sentence[start:end+1], sentence))
            print([(x, round(y, 2)) for x, y in zip(pred, prob)])
        return preds, probs

    def get_frames_probability(self, input_data, frames_lst):
        """
        input_data: input with format [(sentence_1, start_1, end_1), ...]
        frames_lst: filters output to specific frames, has format [[frame_name_1, ...], ...]
        """
        self.dataset.set_input_only(input_data)    
        # Map FN frames to indexes, if defined
        filter_idx_lst = [[FRAME_DICT[f] for f in frames] for frames in frames_lst]
        probs = self.model.get_probabilities_dataset(self.dataset, filter_idx_lst)        
        probs = probs.tolist()

        # Change probabilities to follow the same order as frames_lst
        probs = [[probs[i][j] for j in f_idxs] for i, f_idxs in enumerate(filter_idx_lst)]
        for inputs, frames, prob in zip(input_data, frames_lst, probs):
            sentence, start, end = inputs
            print("Frame probabilities for: '%s' in '%s'" % (sentence[start:end+1], sentence))
            print([(x, round(y, 2)) for x, y in zip(frames, prob)])
        return probs