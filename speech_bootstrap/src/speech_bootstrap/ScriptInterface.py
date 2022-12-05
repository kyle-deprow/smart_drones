#/usr/bin/python

import os
import json
import soundfile as sf

PARTICIPANT_DIR = '/content/speech_bootstrap/participants'
SYNTHESIS_DIR = 'synthesis'
SCRIPT_JSON = 'script.json'

class ScriptInterface():
  def __init__(self, participant: str):
    self._participant_dir = os.path.join(PARTICIPANT_DIR, participant)
    self._synthesis_dir = os.path.join(self._participant_dir, SYNTHESIS_DIR)
    self._part_script = json.load(open(os.path.join(self._participant_dir, SCRIPT_JSON)))
    self._syn_script = json.load(open(os.path.join(self._synthesis_dir, SCRIPT_JSON)))

  def return_syn_phrases(self):
    return self._syn_script['phrase']

  def return_norm_phrases(self):
    return self._part_script['phrase']

  def return_syn_subject_tokens(self):
    return self._syn_script['subject_encoding']

  def return_norm_subject_tokens(self):
    return self._part_script['subject_encoding']

  def load_audio(self, phrase_num):
    return sf.read(os.path.join(self._synthesis_dir, 'phrase{n}.wav'.format(n=phrase_num)))[0]

