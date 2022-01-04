import pathlib
import pygubu
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
import tensorflow as tf
import numpy as np
import os
import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
from tkinter import filedialog as fd
import docx



PROJECT_PATH = pathlib.Path(__file__).parent
PROJECT_UI = PROJECT_PATH / "nd.ui"

# Xử lý
fs = 16000
label_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'XH']
#lấy đường dẫn thư mục hiện tại
current_path = os.getcwd()
#load mô hình đã train
load_model = tf.keras.models.load_model("save.h5")

def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(contents=audio_binary)
  return tf.squeeze(audio, axis=-1)

#thực hiện dự đoán

def get_waveform_audio(file_path):
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform

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

def predic(spectrogram):
    prediction = load_model(spectrogram)
    prediction = prediction.numpy()
    y_pred = np.argmax(prediction)
    return label_name[y_pred]

def pred_file(file_p):
    print('predict file')
    wave = get_waveform_audio(file_p)
    spec = get_spectrogram(wave)
    return predic(spec)

def pred_mic(mic_audio):
    print('mic')
    wave = get_waveform_mic(mic_audio)
    spec = get_spectrogram(wave)
    return predic(spec)


def record_audio_start():
    m = sd.rec(3 * fs, samplerate=fs, channels=1,dtype='float32')
    return m

def record_audio_stop(m):
    print( "Recording Audio")
    sd.wait()
    print( "Audio recording complete.")
    for i in range(len(m)):
        if m[i]>1/10*np.amax(m):
            break
    audio = m[i:i+16000]
    # wav.write("out.wav",fs,audio)
    return audio


#xử lý UI

def select_file():
    filetypes = (
        ('WAV files', '*.wav'),
        ('All files', '*.*')
    )
    filename = fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)
    return filename

def save_text(my_text_box):
    files = [('Text Document', '*.txt')]
    exportFile = fd.asksaveasfile(filetypes = files, defaultextension = files)
    text_file = open(exportFile.name, "w")
    text_file.write(my_text_box.get(1.0, 'end-1c'))
    text_file.close()

def save_docx(my_text_box):
    files = [('Word Document', '*.docx')]
    exportFile = fd.asksaveasfile(filetypes = files, defaultextension = files)
    mydoc = docx.Document()
    mydoc.add_paragraph(my_text_box.get(1.0, 'end-1c'))
    mydoc.save(exportFile.name)




#biến toàn cục
pathwav = None
recording_temp = None
recording_audio = None

# GUI tkinter
class NdApp:
    pathwav =""
    def __init__(self, master=None):
        # build ui
        self.frame1 = ttk.Frame(master)
        self.text1 = tk.Text(self.frame1)
        self.text1.configure(height='10', width='50')
        self.text1.place(anchor='nw', height='150', x='0', y='203')
        self.btnGhi = ttk.Button(self.frame1)
        self.btnGhi.configure(takefocus=False, text='Ghi âm')
        self.btnGhi.place(anchor='nw', relwidth='0.45', x='0', y='50')
        _wcmd = lambda wid='btnGhi': self.btnGhi_clicked(wid)
        self.btnGhi.configure(command=_wcmd)
        self.btnGhi_dung = ttk.Button(self.frame1)
        self.btnGhi_dung.configure(state='normal', takefocus=False, text='Dừng')
        self.btnGhi_dung.place(anchor='nw', x='10', y='80')
        _wcmd = lambda wid='btnGhi_dung': self.btnGhi_dung_clicked(wid)
        self.btnGhi_dung.configure(command=_wcmd)
        self.btnNhanDang = ttk.Button(self.frame1)
        self.btnNhanDang.configure(text='Nhận dạng')
        self.btnNhanDang.place(anchor='nw', relheight='0.14', relwidth='0.84', x='30', y='140')
        _wcmd = lambda wid='btnNhanDang': self.btnNhanDang_clicked(wid)
        self.btnNhanDang.configure(command=_wcmd)
        self.btnGhi_nghe = ttk.Button(self.frame1)
        self.btnGhi_nghe.configure(state='normal', takefocus=False, text='Nghe')
        self.btnGhi_nghe.place(anchor='nw', x='90', y='80')
        _wcmd = lambda wid='btnGhi_nghe': self.btnGhi_nghr_clicked(wid)
        self.btnGhi_nghe.configure(command=_wcmd)
        self.btnTai = ttk.Button(self.frame1)
        self.btnTai.configure(takefocus=False, text='Tải lên')
        self.btnTai.place(anchor='nw', relwidth='0.45', x='220', y='50')
        _wcmd = lambda wid='btnTai': self.btnTai_clicked(wid)
        self.btnTai.configure(command=_wcmd)
        self.btnXuat = ttk.Button(self.frame1)
        self.btnXuat.configure(text='Xuất kết quả')
        self.btnXuat.place(anchor='nw', relheight='0.08', relwidth='0.26', x='290', y='360')
        self.btnXuat.configure(command=self.btnXuat_clicked)
        self.frame1.configure(height='400', width='400')
        self.frame1.pack(side='top')

        # Main widget
        self.mainwindow = self.frame1

    def run(self):
        self.mainwindow.mainloop()

    def btnGhi_clicked(self, widget_id):
        global recording_temp
        print('record_audio_start')
        recording_temp = record_audio_start()
        pass

    def btnGhi_dung_clicked(self, widget_id):
        global recording_audio
        global recording_temp

        if recording_temp is not None:
            print('record_audio_stop')
            recording_audio = record_audio_stop(recording_temp)
            messagebox.showinfo('Thông báo', 'Ghi âm hoàn tất')
        else:
            messagebox.showinfo('Lỗi', 'Không có âm thành đã ghi âm để dừng')
        pass

    def btnNhanDang_clicked(self, widget_id):
        global pathwav
        global recording_audio

        if pathwav is not None:
            kq = pred_file(pathwav)
            if kq == "XH":
                kq = "Xuất Hiện"
            self.text1.insert('end-1c', kq+" ")
            messagebox.showinfo('Thông báo', 'Dự đoán hoàn tất')
        elif recording_audio is not None:
            kq = pred_mic(recording_audio)
            if kq == "XH":
                kq = "Xuất Hiện"
            self.text1.insert('end-1c', kq+" ")
            messagebox.showinfo('Thông báo', 'Dự đoán hoàn tất')
        else:
            messagebox.showinfo('Lỗi', 'Không có âm thành nào để dự đoán')

        pass

    def btnGhi_nghr_clicked(self, widget_id):
        global recording_audio
        if recording_audio is not None:
            # filename = 'out.wav'
            # data, fs = sf.read(filename, dtype='float32')
            sd.play(recording_audio, fs)
            status = sd.wait()
        else:
            messagebox.showinfo('Lỗi', 'Không có âm thành đã ghi âm để nghe')
        pass

    def btnTai_clicked(self, widget_id):
        global pathwav
        pathwav = select_file()
        pass

    def btnXuat_clicked(self):
        save_docx(self.text1)
        messagebox.showinfo('Thông báo', 'Xuất hoàn tất')
        pass



if __name__ == '__main__':
    root = tk.Tk()
    app = NdApp(root)
    app.run()

