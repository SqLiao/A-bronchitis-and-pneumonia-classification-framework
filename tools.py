import os
import librosa
import numpy as np
from logmmse import logmmse_from_file


# Improve signal to noise ratio (SNR)
def ImproveSNR():
    origin_path = ''
    destination_path = ''
    files = os.listdir(origin_path)
    for file in files:
        dis = os.path.join(destination_path, file)
        ori = os.path.join(origin_path, file)
        out = logmmse_from_file(ori, dis)
    return out


def labelDicLoad(p, labelDic):
    print('>>>>>staring \'labelDicLoad\' ' + p + ' ...')
    with open(p, 'r', encoding='utf8') as f:
        while True:
            text = f.readline()
            if text == '': break
            s = text.split(' ')
            temp = s[1].rstrip('\n')
            if len(temp) == 5:
                labelDic[s[0]] = temp[:1]
            else:
                labelDic[s[0]] = temp[:2]
    print('>>>>>\'labelDicLoad\' Finish')


def labelExchangeLoad(labelExDic):
    print('>>>>>staring \'labelExchangeLoad\'...')
    labelExDic['1'] = 1
    labelExDic[1] = '1'
    labelExDic['-1'] = 0
    labelExDic[0] = '-1'
    print('>>>>>\'labelExchangeLoad\' Finish')


# Time shifting
def timeshifting(p, file):
    wav, sr = librosa.load(p, sr=None)
    start_ = int(np.random.uniform(-4800, 4800))
    print('time shift: ', start_)
    if start_ >= 0:
        wav_time_shift = np.r_[wav[start_:], np.random.uniform(-0.001, 0.001, start_)]
    else:
        wav_time_shift = np.r_[np.random.uniform(-0.001, 0.001, -start_), wav[:start_]]
    librosa.output.write_wav('F:/' + file, wav_time_shift, sr)


# Pitch Shift
def pitchShift(p, file):
    wav, sr = librosa.load(p, sr=None)
    pitch_shift1, pitch_shift2 = 0.01, 5.0
    rd1 = np.random.uniform(pitch_shift1, pitch_shift2)
    print('Pitch Shift: ', rd1)
    y_ps = librosa.effects.pitch_shift(wav, sr, n_steps=rd1)
    librosa.output.write_wav('F:/' + file, y_ps, sr)


# Mix background noise
# Randomly choose a slice of background noise then mix it with the audio.
def mixBackgroundNoise(p, file):
    bg_files = os.listdir('./data/_background_noise_/')
    bg_files.remove('README.md')
    bg, sr = librosa.load('./data/_background_noise_/pink_noise.wav', sr=None)  # Add pink noise
    wav, sr = librosa.load(p, sr=None)
    if (bg.shape[0] > wav.shape[0]):
        start_ = np.random.randint(bg.shape[0] - wav.shape[0])
        bg_slice = bg[start_: start_ + wav.shape[0]]
        firstIterm = wav * np.random.uniform(0.8, 1.2)
        secondIterm = bg_slice * np.random.uniform(0, 0.1)
        wav_with_bg = firstIterm + secondIterm
        librosa.output.write_wav('F:/' + file, wav_with_bg, sr)


# Data Augmention
def dataAugmention(p):
    files = os.listdir(p)
    for file in files:
        fileAugPath = p + file

        # Time shifting
        timeshifting(fileAugPath, file)

        # Pitch Shift
        pitchShift(fileAugPath, file)

        # Mix background noise
        mixBackgroundNoise(fileAugPath, file)
        print('--over--')
