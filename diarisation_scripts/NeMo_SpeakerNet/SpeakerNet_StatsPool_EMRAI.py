'''
This script performs speaker diarisation on EMRAI corpus using SpeakerNet from Speaker-NeMo
more info at: https://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels
Trained on voxceleb 1&2 and augmented with MUSAN

Use script with SpekaerNet_Diarisation_EMRAI.yaml
'''

import os
import glob
import json
import warnings
import torch
import pandas as pd
import numpy as np
from math import floor
#from scipy.spatial.distance import cdist
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import h5py

import pickle
import multiprocessing

import pytorch_lightning as pl
#from omegaconf.listconfig import ListConfig
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import hydra
from nemo.collections.asr.models.label_models import ExtractSpeakerEmbeddingsModel

#Pyannote tools used for DER, merging frames, reading rttms etc..
from pyannote.database.util import load_rttm
from pyannote.core import Annotation
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Segment

#from VBx.diarization_lib import l2_norm, cos_similarity, twoGMMcalib_lin, AHC , merge_adjacent_labels
#from VBx.kaldi_utils import read_plda
#from VBx.VB_diarization import VB_diarization

from scipy.special import softmax
from scipy.linalg import eigh

def get_frame_list(start, end, window_size, step_size):
    '''
    :param label_path: path to .lab file
    :param window_size: frame_windows size (.lab is only 10ms)
    :param step_size: step_size for sliding window
    :return: frame label dataframes of shape speakers x frames
    '''
    duration = end-start
    n_increments = floor((duration - window_size) / step_size)
    frames = []
    for i in range(n_increments + 2):
        f_start = round(start+(i * step_size),2)
        f_end = round(f_start + window_size,2)
        frames.append((f_start, f_end))
    return frames


def get_segment_list(file_id, segment_dir):
    segments = [(round(float(line.split()[0]),2), round(float(line.split()[1]),2)) for line in open(segment_dir+'/{}.lab'.format(file_id)) if round(float(line.split()[1]),2) - round(float(line.split()[0]),2) >= 0.4]
    return segments

def write_track_manifest(audio_path, segments, manifest_file):
    with open(manifest_file, 'a') as outfile:
        for i, segment in enumerate(segments):
            start, stop = segment[0], segment[1]
            meta = {"audio_filepath": audio_path, "offset": round(start,2), "duration": round(stop-start,2), "label": 'agent',
                    'segment_id':i}
            json_str = json.dumps(meta)
            outfile.write(json_str)
            outfile.write('\n')


def get_track_embeddings(cfg, id):
    if cfg.audio.num_target_tracks > -1:
        fileids = [line[:line.rfind('\n')] for line in open(cfg.audio.fileids)][0:cfg.audio.num_target_tracks]
        # audio_tracks = glob.glob(cfg.audio.target_path, recursive=True)[:cfg.audio.num_target_tracks]
    else:
        fileids = [line[:line.rfind('\n')] for line in open(cfg.audio.fileids)][0:cfg.audio.num_target_tracks]
        # audio_tracks = glob.glob(cfg.audio.target_path, recursive=True)
    print(fileids)
    os.system('rm -f {}'.format(cfg.audio.track_manifest))

    file_segs = []
    for file in fileids:
        segments = get_segment_list(file, cfg.audio.segments)
        file_segs.append(segments)
        for i, segment in enumerate(segments):
            start, end = segment[0], segment[1]
            frames = get_frame_list(start, end, 0.4, 0.24)
            #print(frames)
            write_track_manifest(audio_path=cfg.audio.target_path+'{}.wav'.format(file), frame_list=frames, segment_id=i, manifest_file=cfg.audio.track_manifest)


    test_config = OmegaConf.create(dict(
        manifest_filepath=cfg.audio.track_manifest,
        sample_rate=16000,
        labels=None,
        batch_size=16,
        shuffle=False,

        embedding_dir=cfg.audio.embedding_dir,
        num_workers=4
    ))

    # GPU access
    cuda = 1 if torch.cuda.is_available() else 0

    # Load model, take a look at ExtractSpeakerEmbeddingsModel function I have changed the code to give each embedding
    # a unique name (test loop)!!!
    model = ExtractSpeakerEmbeddingsModel.from_pretrained(model_name='SpeakerNet_verification')

    model.setup_test_data(test_config)
    trainer = pl.Trainer(gpus=cuda)
    trainer.test(model)
    track_manifest = [json.loads(line.replace('\n', '')) for line in
                      open(cfg.audio.track_manifest)]
    return file_segs, fileids, track_manifest


def cluster_embeddings(segment_embs):
    '''
    Cluster segments from uniform segmentation
    :param track_embedding: The frame-level embeddings from the track to be clustered
    :return: cluster labels
    '''
    #for embedding in track_embedding:
    #    if np.isnan(np.sum(embedding)):
    #        embedding[:] = 0
    # Initialise cluster and fit
    kmeans_cluster = KMeans(n_clusters=2, random_state=5)
    #kmeans_cluster = AgglomerativeClustering(n_clusters=2, affinity='cosine', linkage='average')
    kmeans_cluster.fit_predict(X=segment_embs)

    outputs = []
    for label in kmeans_cluster.labels_:
        outputs.append('cluster_{}'.format(label))
    return outputs


def merge_frames(cluster_outputs, segments):
    annotation = Annotation()
    for i, segment in enumerate(segments):
        annotation[Segment(start=segment[0], end=segment[1])] = cluster_outputs[i]
    annotation = annotation.support()
    return annotation


@hydra.main(config_path='SpeakerNet_StatsPool_EMRAI.yaml')
def main(cfg: DictConfig) -> None:
    if cfg.audio.num_target_tracks > -1:
        fileids = [line[:line.rfind('\n')] for line in open(cfg.audio.fileids)][0:cfg.audio.num_target_tracks]
        # audio_tracks = glob.glob(cfg.audio.target_path, recursive=True)[:cfg.audio.num_target_tracks]
    else:
        fileids = [line[:line.rfind('\n')] for line in open(cfg.audio.fileids)][0:cfg.audio.num_target_tracks]
        # audio_tracks = glob.glob(cfg.audio.target_path, recursive=True)
    print(fileids)

    # GPU access
    cuda = 1 if torch.cuda.is_available() else 0

    # Load model, take a look at ExtractSpeakerEmbeddingsModel function I have changed the code to give each embedding
    # a unique name (test loop)!!!
    model = ExtractSpeakerEmbeddingsModel.from_pretrained(model_name='SpeakerNet_verification')

    # remove the track_manifest if it exists
    # probably better to do with os.remove but this is easier
    os.system('rm -f {}'.format(cfg.audio.track_manifest))
    #file_segs, fileids, track_manifest = get_track_embeddings(cfg)


    for i, id in enumerate(fileids):
        segments = get_segment_list(id, cfg.audio.segments)
        #file_segs.append(segments)
        os.system('rm -f {}'.format(cfg.audio.track_manifest))
        write_track_manifest(audio_path=cfg.audio.target_path + '{}.wav'.format(id), segments=segments,
                             manifest_file=cfg.audio.track_manifest)

        test_config = OmegaConf.create(dict(
            manifest_filepath=cfg.audio.track_manifest,
            sample_rate=16000,
            labels=None,
            batch_size=16,
            shuffle=False,
            embedding_dir=cfg.audio.embedding_dir,
            num_workers=4
        ))

        model.setup_test_data(test_config)
        trainer = pl.Trainer(gpus=cuda)
        trainer.test(model)
        track_manifest = [json.loads(line.replace('\n', '')) for line in open(cfg.audio.track_manifest)]

        segment_embs = []
        with open(os.path.join(cfg.audio.embedding_dir, 'embeddings/track_manifest_embeddings.pkl'), 'rb') as f:
            data = pickle.load(f).items()
            all_track_embeddings = [emb for _, emb in data]
            print(len(all_track_embeddings))
        for k, segment in enumerate(segments):
            indices = [track_manifest.index(item) for item in track_manifest if item['audio_filepath'] == cfg.audio.target_path+id+'.wav' and int(item['segment_id']) == k]
            #print(indices)
            #/np.linalg.norm(x=emb, ord=2, keepdims=True)
            seg_emb = [emb for emb in all_track_embeddings[indices[0]]][0]
            #print(len(seg_emb))
            #seg_emb = np.mean(seg_emb, axis=0, keepdims=True)
            #seg_emb = seg_emb/np.linalg.norm(x=seg_emb, ord=2, keepdims=True)
            segment_embs.append(seg_emb.flatten())
        cluster_outputs = cluster_embeddings(segment_embs)
        annotation = merge_frames(cluster_outputs, segments)
        os.system('mkdir -p {}'.format(cfg.audio.out_rttm_path))
        with open(os.path.join(os.path.join(cfg.audio.out_rttm_path, '{}.rttm'.format(id))), 'w') as file:
            annotation.write_rttm(file)




if __name__ == '__main__':
    main()
