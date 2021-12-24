import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.models import load_model

from IPython import display
# from training import preprocess_dataset


def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(contents=audio_binary)
  return tf.squeeze(audio, axis=-1)

# def decode_audio(audio):
#   au_wav = tf.convert_to_tensor(audio, dtype=tf.float32)
#   au_wav = tf.reshape(au_wav,[au_wav.shape[0],1])
#   waveform = tf.squeeze(au_wav, axis=-1)
#   return waveform


def get_waveform(file_path):
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform



def preprocess_dataset(files):
  AUTOTUNE = tf.data.AUTOTUNE
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(
      map_func=get_waveform,
      num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      map_func=get_waveform,
      num_parallel_calls=AUTOTUNE)
  return output_ds

DATASET_PATH = 'data/test'
data_dir = pathlib.Path(DATASET_PATH)

commands = np.array([0,1,2,3,4,5,6,7,8,9])
print('Commands:', commands)

#LoadModel
model = load_model('save.h5')


## Run inference on an audio file

sample_file = 'data\test\5_thuc_25.wav'

sample_ds = preprocess_dataset(sample_file)

# for spectrogram, label in sample_ds.batch(1):
#   prediction = model(spectrogram)
#   plt.bar(commands, tf.nn.softmax(prediction[0]))
#   plt.title(f'Predictions for "{commands[label[0]]}"')
#   plt.show()