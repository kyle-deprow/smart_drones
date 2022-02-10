#/usr/bin/python

import numpy as np
import json

def split_data(path):
  sentences = []
  encodings = []
  json_dict = json.load(open(path))
  for configuration, internal_dict in json_dict.items():
    sentences.extend(internal_dict['phrase'])
    encoding = internal_dict['subject_encoding']
    encodings.extend([encoding]*len(internal_dict['phrase']))
  return sentences, encodings

def train_model(X, y)

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', required=True, type=str)
  args = parser.parse_args()
  split_data(args.data_path)

