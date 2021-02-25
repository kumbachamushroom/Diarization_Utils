import torch
import numpy as np
import librosa
import torchaudio
import matplotlib
import math
import os
import subprocess
import glob
import sox
import random
import hydra
from omegaconf import DictConfig
from math import floor
import soundfile as sf


def get_RMS(np_signal):
    return math.sqrt(np.mean(np_signal**2))

def get_SNR(RMS_signal, RMS_noise):
    return 10*math.log((RMS_signal**2/RMS_noise**2),10)


def get_RMS_required(RMS_signal, SNR):
    return math.sqrt(RMS_signal**2/10**(SNR/10))

def attenuate_noise(np_noise, RMS_noise, RMS_required):
    coeff = RMS_required/RMS_noise
    return np_noise * coeff

def add_signals(track, noise):
    if len(track) < len(noise):
        c = noise.copy()
        #print(c)
        c[:len(track)] += track
        #print(c)
        return c
    else:
        c = track.copy()
        #print(c)
        c[:len(noise)] += noise
        #print(c)
        return c

def overlay_noise(input_list, out_dir, SNR_list, noise, noise_duration):
    for SNR in SNR_list:
        print(SNR)
        os.system('mkdir -p {}'.format(out_dir+'/SNR_{}'.format(SNR)))
        for input_filename in input_list:
            track, sr = librosa.load(input_filename)
            track_duration = librosa.get_duration(y=track, sr=sr)
            print('track duration is {}'.format(track_duration))
            start = random.randrange(0,floor(noise_duration)-floor(track_duration))
            print('start is {}'.format(start))
            noise_proc = noise[start*sr:floor(track_duration)*sr]
            print(len(noise_proc))
            RMS_signal = get_RMS(track)
            RMS_noise = get_RMS(noise_proc)
            RMS_required = get_RMS_required(RMS_signal, SNR)
            noise_proc = attenuate_noise(noise_proc, RMS_noise, RMS_required)
            print(noise_proc)
            #sf.write(file=os.path.join(out_dir,'SNR_{}'.format(SNR),input_filename[input_filename.rfind('/')+1:]),data=add_signals(track,noise), samplerate=8000)
            librosa.output.write_wav(path=os.path.join(out_dir,'SNR_{}'.format(SNR),input_filename[input_filename.rfind('/')+1:]), y=add_signals(track, noise_proc), sr=sr)


@hydra.main(config_path="./Add_MUSAN_Noise.yaml")
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())
    input_list = glob.glob(cfg.audio.input+'/*', recursive=True)
    os.system('mkdir -p {}'.format(cfg.audio.output))
    noise, sr = librosa.load(cfg.augmentation.noise_dir)
    print(noise)
    noise_duration = librosa.get_duration(y=noise, sr=sr)
    print(noise_duration)
    overlay_noise(input_list=input_list,out_dir=cfg.audio.output,SNR_list=cfg.augmentation.SNR, noise=noise, noise_duration=noise_duration)

if __name__ == '__main__':
    main()

