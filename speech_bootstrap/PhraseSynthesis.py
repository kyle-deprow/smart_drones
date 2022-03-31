#/usr/bin/python

import os
import sys

import numpy as np
import json
from pydub import AudioSegment
import librosa
import soundfile as sf

import torch

from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.symbols import make_symbols, phonemes, symbols
try:
  from TTS.utils.audio import AudioProcessor
except:
  from TTS.utils.audio import AudioProcessor

from TTS.tts.models import setup_model
from TTS.config import load_config
from TTS.tts.models.vits import *
from TTS.tts.utils.speakers import SpeakerManager

from smart_drones.speech_bootstrap.util import load_subject_data, generate_participant_scripts

# model vars 
MODEL_PATH = 'content/best_model.pth.tar'
CONFIG_PATH = 'config.json'
TTS_LANGUAGES = 'content/language_ids.json'
TTS_SPEAKERS = 'content/speakers.json'
USE_CUDA = torch.cuda.is_available()
CONFIG_SE_PATH = 'config_SE.json'
CHECKPOINT_SE_PATH = 'SE_checkpoint.pth.tar'
WAV_EXT = '.wav'
SYNTHESIS_DIR = 'synthesis'
NORMALIZE_EXT = '_norm'
TEMP_EXT = '_temp'

class PhraseSynthesis():
  def __init__(self, ttsmodel_dir: str, participant_dir: str):
    self._ttsmodel_dir = ttsmodel_dir
    self._participant_dir = participant_dir
    self._load_ttsmodel()
    self._load_speaker_encoder()

    self._reference_wav_files = []
    for fpath in os.listdir(participant_dir):
      if fpath.endswith(WAV_EXT) and NORMALIZE_EXT not in fpath:
        self._reference_wav_files.append(os.path.join(participant_dir, fpath))

    self._synthesis_dir = os.path.join(participant_dir, SYNTHESIS_DIR)
    if not os.path.isdir(self._synthesis_dir):
      os.mkdir(self._synthesis_dir)

    self._determine_reference_embeddings()

  def _load_ttsmodel(self) -> None: wget
    # load the config
    self._tts_config = load_config(os.path.join(self._ttsmodel_dir, CONFIG_PATH))

    # load the audio processor
    self._audio_proc = AudioProcessor(**self._tts_config.audio)

    self._tts_config.model_args['d_vector_file'] = TTS_SPEAKERS
    self._tts_config.model_args['use_speaker_encoder_as_loss'] = False

    self._model = setup_model(self._tts_config)
    self._model.language_manager.set_language_ids_from_file(TTS_LANGUAGES)
    cp = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    # remove speaker encoder
    model_weights = cp['model'].copy()
    for key in list(model_weights.keys()):
      if "speaker_encoder" in key:
        del model_weights[key]
    self._model.load_state_dict(model_weights)
    self._model.eval()
    if USE_CUDA:
        self._model = self._model.cuda()

  def _load_speaker_encoder(self):
    self._speaker_encoder = SpeakerManager(
        encoder_model_path=os.path.join(self._ttsmodel_dir, CHECKPOINT_SE_PATH),
        encoder_config_path=os.path.join(self._ttsmodel_dir, CONFIG_SE_PATH),
        use_cuda=USE_CUDA)

  def _determine_reference_embeddings(self):
    norm_wav_files = []
    for sample in self._reference_wav_files:
      norm_wav_file = sample.split('.wav')[0] + NORMALIZE_EXT + '.wav'
      if not os.path.exists(norm_wav_file):
          #read wav data and trim silence
          temp_wav_file = sample.split('.wav')[0] + TEMP_EXT + '.wav'
          audio, sr = librosa.load(sample, sr= 16000, mono=True)
          clip = librosa.effects.trim(audio, top_db=10)
          sf.write(temp_wav_file, clip[0], sr)

          cmd_str = 'ffmpeg-normalize {s} -nt rms -t=-27 -o {os} -ar 16000 -f -c:a pcm_s16le'.format(s=temp_wav_file, os=norm_wav_file)
          os.system(cmd_str)
          os.remove(temp_wav_file)
      norm_wav_files.append(norm_wav_file)
    self._reference_emb = self._speaker_encoder.compute_d_vector_from_clip(norm_wav_files)

  def synthesize_phrases(self, data: dict):
    self._model.length_scale = 1  # scaler for the duration predictor. The larger it is, the slower the speech.
    self._model.inference_noise_scale = 0.3 # defines the noise variance applied to the random z vector at inference.
    self._model.inference_noise_scale_dp = 0.3 # defines the noise variance applied to the duration predictor z vector at inference.

    synthesis_directory = os.path.join(self._participant_dir, SYNTHESIS_DIR)
    for i,text in enumerate(data['phrase']):
      wav, alignment, _, _ = synthesis(
                            self._model,
                            text,
                            self._tts_config,
                            "cuda" in str(next(self._model.parameters()).device),
                            self._audio_proc,
                            speaker_id=None,
                            d_vector=self._reference_emb,
                            style_wav=None,
                            language_id=0,
                            enable_eos_bos_chars=self._tts_config.enable_eos_bos_chars,
                            use_griffin_lim=True,
                            do_trim_silence=False,
                        ).values()
      file_name = 'phrase{n}.wav'.format(n=i+1)
      out_path = os.path.join(synthesis_directory, file_name)
      print(' > Saving output to {}'.format(out_path))
      self._audio_proc.save_wav(wav, out_path)
    generate_participant_scripts(participant_dir=synthesis_directory, data=data)

if __name__ == '__main__':
  syn = PhraseSynthesis('./content', './smart_drones/speech_bootstrap/participants/participant1')
  data = load_subject_data('./smart_drones/speech_bootstrap/phrases.json', 9000, 1000)
  syn.synthesize_phrases(data)
  

