#/usr/bin/python

import os
import sys

import numpy as np
import json
from pydub import AudioSegment
import librosa

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

from smart_drones.speech_bootstrap.util import load_subject_data

# model vars 
MODEL_PATH = 'best_model.pth.tar'
CONFIG_PATH = 'config.json'
TTS_LANGUAGES = "language_ids.json"
TTS_SPEAKERS = "speakers.json"
USE_CUDA = torch.cuda.is_available()
CONFIG_SE_PATH = "config_se.json"
CHECKPOINT_SE_PATH = "SE_checkpoint.pth.tar"
WAV_EXT = '.wav'

class PhraseSynthesize():
  def __init__(self, zeroshot_dir: str, participant_dir: str):
    self._zeroshot_dir = zeroshot_dir
    self._participant_dir = participant_dir
    self._load_zeroshot()
    self._load_speaker_encoder()

    self._reference_wav_files = []
    for fpath in os.listdir(participant_path):
      if fpath.endswith(WAV_EXT):
        self._reference_wav_files.append(os.path.join(participant_path, fpath))

    self._synthesis_dir = os.path.join(participant_dir, SYNTHESIS_DIR)
    if not os.path.isdir(self._synthesis_dir):
      os.mkdir(self._synthesis_dir)

  def _load_zeroshot(self) -> None:
    # load the config
    self._tts_config = load_config(os.path.join(self._zeroshot_dir, CONFIG_PATH))

    # load the audio processor
    self._audio_proc = AudioProcessor(**C.audio)

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

  def _load_speaker_encoder(self, zeroshot_dir: str):
    self._speaker_encoder = SpeakerManager(
        encoder_model_path=os.path.join(self._zeroshot_dir, CHECKPOINT_SE_PATH),
        encoder_config_path=os.path.join(self._zeroshot_dir, CONFIG_SE_PATH),
        use_cuda=USE_CUDA)

  def _determine_reference_embeddings(self, data: dict):
    for sample in self._reference_wav_files:
        !ffmpeg-normalize $sample -nt rms -t=-27 -o $sample -ar 16000 -f
    self._reference_emb = self._speaker_encoder.compute_d_vector_from_clip(self._reference_wav_files)

  def synthesize_phrases(self, data: dict):
    model.length_scale = 1  # scaler for the duration predictor. The larger it is, the slower the speech.
    model.inference_noise_scale = 0.3 # defines the noise variance applied to the random z vector at inference.
    model.inference_noise_scale_dp = 0.3 # defines the noise variance applied to the duration predictor z vector at inference.
    
    print(' > text: {}'.format(text))
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
                        enable_eos_bos_chars=C.enable_eos_bos_chars,
                        use_griffin_lim=True,
                        do_trim_silence=False,
                    ).values()
    print("Generated Audio")
    file_name = text.replace(" ", "_")
    file_name = file_name.translate(str.maketrans('', '', string.punctuation.replace('_', ''))) + '.wav'
    out_path = os.path.join(OUT_PATH, file_name)
    print(' > Saving output to {}'.format(out_path))
    self._audio_proc.save_wav(wav, out_path)

