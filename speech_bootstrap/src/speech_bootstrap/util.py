#/usr/bin/python

import os
import numpy as np
import json

SCRIPT_FILE = 'script.json'
def load_subject_data(path: str, n_phrases: int, n_dummy: int) -> dict:
  sentences = []
  sub_encodings = []
  task_encodings = []

  sentences_dummy = []
  sub_encodings_dummy = []
  task_encodings_dummy = []
  json_dict = json.load(open(path))
  for configuration, internal_dict in json_dict.items():
    if configuration == 'dummy_data': 
      sentences_dummy.extend(internal_dict['phrase'])
      sub_encoding_dummy = internal_dict['subject_encoding']
      task_encoding_dummy = internal_dict['task_encoding']
      sub_encodings_dummy.extend([sub_encoding_dummy]*len(internal_dict['phrase']))
      task_encodings_dummy.extend([task_encoding_dummy]*len(internal_dict['phrase']))
    else:
      sentences.extend(internal_dict['phrase'])
      sub_encoding = internal_dict['subject_encoding']
      task_encoding = internal_dict['task_encoding']
      sub_encodings.extend([sub_encoding]*len(internal_dict['phrase']))
      task_encodings.extend([task_encoding]*len(internal_dict['phrase']))
  phrase_ind = np.random.randint(0, len(sentences), n_phrases).tolist()
  dummy_ind = np.random.randint(0, len(sentences_dummy), n_dummy).tolist()
  rand_sentences = [sentences[i] for i in phrase_ind]
  rand_sub_encodings = [sub_encodings[i] for i in phrase_ind]
  rand_task_encodings = [task_encodings[i] for i in phrase_ind]
  rand_sentences.extend([sentences_dummy[i] for i in dummy_ind])
  rand_sub_encodings.extend([sub_encodings_dummy[i] for i in dummy_ind])
  rand_task_encodings.extend([task_encodings_dummy[i] for i in dummy_ind])
  return {'phrase': rand_sentences, 'subject_encoding': rand_sub_encodings, 'task_encoding': rand_task_encodings}

def generate_participant_scripts(participant_dir: str, database_path=None, n_phrases=0, n_dummy=0, data=None) -> None:
  if not data:
      data = load_subject_data(database_path, n_phrases, n_dummy)
  script_file = os.path.join(participant_dir, SCRIPT_FILE)
  json.dump(data, open(script_file, 'w+'), indent=4)

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--utility', required=False, type=str)
  parser.add_argument('--phrase_path', required=False, type=str)
  parser.add_argument('--participant_path', required=False, type=str)
  parser.add_argument('--nphrases', required=False, type=int)
  parser.add_argument('--ndummy', required=False, type=int)
  args = parser.parse_args()
  if args.utility == 'generate_script':
    generate_participant_scripts(args.participant_path, args.phrase_path, args.nphrases, args.ndummy)
