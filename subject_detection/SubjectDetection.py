#/usr/bin/python

import numpy as np
import pandas as pd
import torch
import transformers as ppb # pytorch transformers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
    
class SubjectDetectionModel():
  def __init__(self):
  self.model_class, self.tokenizer_class, self.pretrained_weights =
    (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

  # Load pretrained model/tokenizer
  self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
  self.model = model_class.from_pretrained(pretrained_weights)

  def tokenize(self, input_sentences):
    return [tokenizer.encode(x, add_special_tokens=True) for x in input_sentences]

  def train_test_split(self, X, y, train_percentage):
    assert len(X) == len(y)
    size = len(X)
    return X[:size], y[:size], X[size:], y[size:]

  def train(self, X, y):
    max_len = 0
    for i in X:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    attention_mask = np.where(padded != 0, 1, 0)
    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)
    features = last_hidden_states[0][:,0,:].numpy()
    labels = batch_1[1]
    Xtrain, ytrain, Xtest, ytest = self.train_test_split(X, y, 0.8)


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', required=True, type=str)
  args = parser.parse_args()
  split_data(args.data_path)

