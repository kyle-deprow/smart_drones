#/usr/bin/python

import os
import numpy as np
import json
import itertools
from num2words import num2words
import nltk

def subject_generation(sub_comb, name_list):
  length = len(sub_comb)
  def return_start_combination(name, start_sub):
    return name + num2words(start_sub)

  def return_middle_combinations(name, middle_sub):
    return [', ' + name + num2words(middle_sub),\
            ', ' + num2words(middle_sub)]

  def return_end_combinations(name, end_sub):
    return [' and ' + name + num2words(end_sub),\
            ', ' + name + num2words(end_sub),\
            ' and ' + num2words(end_sub),\
            ', ' + num2words(end_sub)]

  def return_through_combinations(name, first_sub, last_sub):
    return [name + num2words(first_sub) + ' through ' + name + num2words(last_sub),\
            name + num2words(first_sub) + ' through ' + num2words(last_sub)]

  rvalue = []
  for name in name_list:
    string = [return_start_combination(name, sub_comb[0])]
    for i in range(1, length - 1):
      string = [string[0] + comb for comb in return_middle_combinations(name, sub_comb[i])]
    if len(sub_comb) > 1:
      string = [string_start + string_end for string_start in string for string_end in return_end_combinations(name, sub_comb[-1])]
    rvalue.extend(string)
  # Check special case where tuple is ordered and you say subject through subject
  # ie drone 1 through 3
  if sub_comb == tuple(sorted(sub_comb)) and len(sub_comb) > 1:
    rvalue.extend(return_through_combinations(name, sub_comb[0], sub_comb[-1]))
  return rvalue

def encode_subjects(subf, subl, name_list):
  sub_encoding = [0]*(subl - subf + 1)
  sub_list = np.arange(subf, subl + 1)
  sub_comb = {}

  for L in range(1, len(sub_list)+1):
    for comb in itertools.permutations(sub_list, L):
      sub_comb[comb] = {'subjects': None, 'encoding':np.zeros(subl - subf + 1)}

  for combination, value in sub_comb.items():
    sub_comb[combination]['subjects'] = subject_generation(combination, name_list)
    sub_comb[combination]['encoding'][np.array(combination) - 1] = 1

  return sub_comb

def encode_tasks(move_points, move_phrases, shape_tasks, shape_phrases):
  mapping = {task: i for i, task in enumerate(move_points + shape_tasks)}
  rvalue = {task: {'task': None, 'encoding': None} for task in mapping.keys()}
  for pt in move_points:
    rvalue[pt]['task'] = []
    rvalue[pt]['encoding'] = np.zeros(len(mapping))
    rvalue[pt]['encoding'][mapping[pt]] = 1
    for verb in move_phrases:
      rvalue[pt]['task'] += [verb + pt]
  
  for shape in shape_tasks:
    rvalue[shape]['task'] = []
    rvalue[shape]['encoding'] = np.zeros(len(mapping))
    rvalue[shape]['encoding'][mapping[shape]] = 1
    for verb in shape_phrases:
      rvalue[shape]['task'] += [verb + shape]
  return rvalue

def generate_dummy_phrases(input_dir, n_phrases):
  sentences = []
  input_files = []
  for (dirpath, dirnames, filenames) in os.walk(input_dir):
    input_files.extend(filenames)

  for infile in input_files:
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    fp = open(os.path.join(input_dir, infile))
    data = fp.read()
    for sentence in tokenizer.tokenize(data):
      if 5 < len(sentence.split(' ')) < 13:
        sentences.append(sentence)
    if len(sentences) > n_phrases:
      break
  return sentences

def generate_phrases(json_input, json_output):
  json_dict = json.load(open(json_input))
  subjects = encode_subjects(
    int(min(json_dict['drone_numbers'])),
    int(max(json_dict['drone_numbers'])),
    json_dict["drone_names"])
  tasks = encode_tasks(
    json_dict['move_points'],
    json_dict['move_phrases'],
    json_dict['shape_tasks'],
    json_dict['shape_phrases'])
  composite = {}
  n_phrases = 0
  for subject, subject_dict in subjects.items():
    for task, task_dict in tasks.items():
      subject_str = [str(sub) for sub in subject]
      scenario_name = ','.join(subject_str + [task])
      composite[scenario_name] =\
        {'phrase': [subject_phrase + ' ' + task_phrase for subject_phrase in subject_dict['subjects']
          for task_phrase in task_dict['task']],\
         'subject_encoding': list(np.concatenate([subject_dict['encoding'], [0]])),\
         'task_encoding': list(np.concatenate([task_dict['encoding'], [0]]))}
      n_phrases += len(composite[scenario_name]['phrase'])
  
  composite['dummy_data'] = {}
  composite['dummy_data']['phrase'] = generate_dummy_phrases(json_dict['dummy_data_directory'], n_phrases*0.5)
  composite['dummy_data']['subject_encoding'] = list(np.concatenate([np.zeros(len(composite[scenario_name]['subject_encoding']) - 1), [1]]))
  composite['dummy_data']['task_encoding'] = list(np.concatenate([np.zeros(len(composite[scenario_name]['task_encoding']) - 1), [1]]))

  json.dump(composite, open(json_output, 'w'), indent=6)


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--config_path', required=True, type=str)
  parser.add_argument('--output', required=True, type=str)
  args = parser.parse_args()
  generate_phrases(args.config_path, args.output)

