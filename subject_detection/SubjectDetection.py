#/usr/bin/python

import os
import numpy as np

import random
import tensorflow as tf
import tensorflow_hub as hub
from wav2vec2 import Wav2Vec2Config
from wav2vec2 import Wav2Vec2Processor
from smart_drones.speech_bootstrap.ScriptInterface import ScriptInterface
from wav2vec2 import CTCLoss


AUDIO_MAXLEN = 246000
LABEL_MAXLEN = 256
BATCH_SIZE = 8

class SubjectDetectionModel():
  def __init__(self):
    self.tokenizer = Wav2Vec2Processor(is_tokenizer=True)
    self.processor = Wav2Vec2Processor(is_tokenizer=False)

  def build_model(self):
    config = Wav2Vec2Config()
    pretrained_layer = hub.KerasLayer("https://tfhub.dev/vasudevgupta7/wav2vec2/1", trainable=True)
    inputs = tf.keras.Input(shape=(AUDIO_MAXLEN,))
    hidden_states = pretrained_layer(inputs)
    outputs = tf.keras.layers.Dense(config.vocab_size)(hidden_states)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    script_if = ScriptInterface('participant1')
    tokens = script_if.return_syn_subject_tokens()
    phrases = script_if.return_syn_phrases()
    sample_length = len(tokens)
    self.samples = [(script_if.load_audio(i+1), phrase) for i, phrase in enumerate(phrases)]

    output_signature = (
      tf.TensorSpec(shape=(None),  dtype=tf.float32),
          tf.TensorSpec(shape=(None), dtype=tf.int32),
    )
    dataset = tf.data.Dataset.from_generator(self.inputs_generator, output_signature=output_signature)
    dataset = dataset.shuffle(sample_length, seed=42)
    dataset = dataset.padded_batch(sample_length, padded_shapes=(AUDIO_MAXLEN, LABEL_MAXLEN), padding_values=(0.0, 0))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    num_train_batches = 10
    num_val_batches = 4

    train_dataset = dataset.take(num_train_batches)
    val_dataset = dataset.skip(num_train_batches).take(num_val_batches)
    loss_fn = CTCLoss(config, (sample_length, AUDIO_MAXLEN), division_factor=sample_length)
    optimizer = tf.keras.optimizers.Adam(5e-5)
    model.compile(optimizer, loss=loss_fn)
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=3)
    history.history

  def preprocess_text(self, text):
    label = self.tokenizer(text)
    return tf.constant(label, dtype=tf.int32)

  def preprocess_speech(self, audio):
    audio = tf.constant(audio, dtype=tf.float32)
    return self.processor(tf.transpose(audio))

  def inputs_generator(self):
    for speech, text in self.samples:
      yield self.preprocess_speech(speech), self.preprocess_text(text)

if __name__ == '__main__':
  model = SubjectDetectionModel()
  model.build_model()

