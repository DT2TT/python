#!/usr/bin/env python
# coding: utf-8

# In[1]:
import tensorflow as tf
import numpy as np
import pathlib
import os
import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
# In[2]:
fs = 16000
label_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#lấy đường dẫn thư mục hiện tại
current_path = os.getcwd()
#load mô hình đã train
load_model = tf.keras.models.load_model("save.h5")

# In[8]:
def get_waveform_audio(file_path):
    audio = librosa.load(file_path, sr = 16000)
    audio = tf.convert_to_tensor(audio[0], dtype=tf.float32)
    audio = tf.reshape(audio,[audio.shape[0],1])
    waveform = tf.squeeze(audio, axis=-1)
    return waveform
def record_audio():    
    input("Press Enter to start record...")
    m = sd.rec(3 * fs, samplerate=fs, channels=1,dtype='float32')
    print( "Recording Audio")
    sd.wait()
    print( "Audio recording complete.")
    for i in range(len(m)):
        if m[i]>1/10*np.amax(m):
            break
    audio = m[i:i+16000]
    return audio
def get_waveform_mic(audio):
    au_wav = tf.convert_to_tensor(audio, dtype=tf.float32)
    au_wav = tf.reshape(au_wav,[au_wav.shape[0],1])
    waveform = tf.squeeze(au_wav, axis=-1)
    return waveform

def get_spectrogram(waveform):
  input_len = 16000
  waveform = waveform[:input_len]
  zero_padding = tf.zeros(
      [16000] - tf.shape(waveform),
      dtype=tf.float32)
  waveform = tf.cast(waveform, dtype=tf.float32)
  equal_length = tf.concat([waveform, zero_padding], 0)
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)
  spectrogram = tf.reshape(spectrogram,[1,124,129,1])
  return spectrogram

def get_spectrogram1(waveform):
    print(tf.shape(waveform))
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.cast(waveform, tf.float32)
    audio_equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        audio_equal_length, frame_length=255, frame_step=127)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.reshape(spectrogram,[1,124,129,1])
    return spectrogram


def predic(spectrogram):
    prediction = load_model(spectrogram)
    prediction = prediction.numpy()
    y_pred = np.argmax(prediction)
    print('The word is: ',label_name[y_pred],' (',"{0:.2f}%".format(prediction[0][y_pred]*100),')')
def pred_file():
    print('file')
    file_p = input('Enter your file name: ')
    file_p = os.path.join(current_path,file_p)
    wave = get_waveform_audio(file_p)
    spec = get_spectrogram(wave)
    predic(spec)
def pred_mic():
    print('mic')
    mic_audio = record_audio()
    wave = get_waveform_mic(mic_audio)
    spec = get_spectrogram(wave)
    predic(spec)


# In[9]:
#chương trình chính
run = 1
while (run == 1) :
    print('Please select: ')
    print('1. Input a file name')
    print('2. Record with computer micro')
    s = input('++>Enter your selection: ')
    if s == '1':
        pred_file()
    elif s == '2':
        pred_mic()
    else :
        print('Invalid input, please try again')
    s = input('++>Do you want to continue(y/n): ')
    if s.find('n') != -1 :
        run = 0


