'''
This script performs speaker diarisation on EMRAI corpus using SincTDNN from pyannote-audio
more info at: https://github.com/juanmc2005/SpeakerEmbeddingLossComparison and https://arxiv.org/pdf/2003.14021.pdf
Trained on voxceleb 1&2 and augmented with MUSAN

Use script with SincTDNN_Diarisation_EMRAI.yaml
'''

import os
import glob
import json
import warnings
import torch
import pandas as pd
import numpy as np
from math import floor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import pickle

import pytorch_lightning as pl
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import hydra


#Pyannote tools used for DER, merging frames, reading rttms etc..
from pyannote.database.util import load_rttm
from pyannote.core import Annotation
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Segment

def label_frames(label_path, window_size, step_size):
    '''
    :param label_path: path to .lab file
    :param window_size: frame_windows size (.lab is only 10ms)
    :param step_size: step_size for sliding window
    :return: frame label dataframes of shape speakers x frames
    '''
    def get_common_label(List):
        return max(set(List), key=List.count)
    labels = [int(line) for line in open(label_path)]
    duration = len(labels)*0.01
    n_increments = floor((duration - window_size)/ step_size)
    frame_list = []
    frame_labels = []
    for i in range(n_increments + 2):
        start_time = i * step_size
        stop_time = start_time + window_size
        frame_list.append((start_time, stop_time))

    for frame in frame_list:
        start_time, stop_time = int(frame[0]/0.01), int(frame[1]/0.01)
        try:
            frame_label = get_common_label(labels[start_time:stop_time])
            frame_labels.append(frame_label)
        except:
            frame_label.append(0)
            None


    return frame_list, frame_labels

def merge_frames(outputs, frame_list, frame_labels):
    annotation = Annotation()
    seg_start = 0
    active_spk = 3
    #print(frame_labels)
    for i, frame in enumerate(frame_list):
        if frame_labels[i] != 0:
            annotation[Segment(start=float(frame[0]), end=float(frame[1]))] = outputs[i]
    annotation.support(collar=0)
    #print(annotation.get_timeline())
    return annotation

def get_track_embeddings(model, frame_list, path):
    embeddings = []
    duration = round(frame_list[-1][1],1)
    for i in range(len(frame_list)):
        start, stop = round(frame_list[i][0],1), round(frame_list[i][1],1)
        try:
            excerpt = Segment(start=start, end=stop)
            embedding = np.mean(model.crop({'audio': path, 'duration': duration}, segment=excerpt), axis=0, keepdims=True)
            embeddings.append(embedding)
        except:
            embeddings.append(np.zeros(shape=(1,512), dtype=float))
            print('could not embed  ',start, stop)
    return np.concatenate(embeddings, axis=0)


@hydra.main(config_path="SincTDNN_Diarisation_EMRAI.yaml")
def main(cfg: DictConfig) -> None:
    #load SincTDN model
    model = torch.hub.load('pyannote/pyannote-audio', 'emb_voxceleb')
    # load track names
    if cfg.audio.num_target_tracks == -1:
        fileids = [line[:line.rfind('\n')] for line in open(cfg.audio.fileids)]
    else:
        fileids = [line[:line.rfind('\n')] for line in open(cfg.audio.fileids)][0:cfg.audio.num_target_tracks]
    for window_length in cfg.audio.window_length:
        for step_length in cfg.audio.step_length:
            for track in fileids:
                label_path = cfg.audio.label_path + track + '.labs'
                frame_list, frame_labels = label_frames(label_path=label_path,
                                                        window_size=window_length,
                                                        step_size=float(window_length * step_length))
                embeddings = get_track_embeddings(model=model, frame_list=frame_list, path=cfg.audio.target_path+track+'.wav')
if __name__ == '__main__':
    main()
