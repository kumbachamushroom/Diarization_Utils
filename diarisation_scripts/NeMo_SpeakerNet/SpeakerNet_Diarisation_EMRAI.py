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

from VBx.diarization_lib import l2_norm, cos_similarity, twoGMMcalib_lin, AHC , merge_adjacent_labels
from VBx.kaldi_utils import read_plda
from VBx.VB_diarization import VB_diarization

from scipy.special import softmax
from scipy.linalg import eigh

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
            frame_labels.append(0)
            #None


    return frame_list, frame_labels



def write_track_manifest(audio_path, frame_list, manifest_file, window_length, step_length):
    with open(manifest_file, 'a') as outfile:
        for i in range(len(frame_list)):
            start, stop = round(frame_list[i][0],1), round(frame_list[i][1],1)
            meta = {"audio_filepath":audio_path, "offset":start, "duration":window_length, "label":'agent',
                    'window_length':window_length, 'step_length':step_length}
            json_str = json.dumps(meta)
            outfile.write(json_str)
            outfile.write('\n')




def cluster_embeddings(track_embedding):
    '''
    Cluster segments from uniform segmentation
    :param track_embedding: The frame-level embeddings from the track to be clustered
    :return: cluster labels
    '''

    # Initialise cluster and fit
    kmeans_cluster = KMeans(n_clusters=2, random_state=5)
    #kmeans_cluster = AgglomerativeClustering(n_clusters=2, affinity='cosine', linkage='average')
    kmeans_cluster.fit_predict(X=track_embedding)

    outputs = []
    for label in kmeans_cluster.labels_:
        outputs.append('cluster_{}'.format(label))
    return outputs

def cluster_PCA(cluster_outputs, labels):
    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(cluster_outputs)
    pca_one = pca_results[:,0]
    pca_two = pca_results[:,1]
    pca_three = pca_results[:,2]
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    ax.scatter(xs = pca_one,ys=pca_two, zs=pca_three,c=labels, cmap='tab10')
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    ax.set_title('PCA-3 decomposition for 512-dimensional embedding')
    plt.show()

    df = pd.DataFrame()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(cluster_outputs)
    df['tsne-one'] = tsne_results[:,0]
    df['tsne-two'] = tsne_results[:,1]

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x='tsne-one',
        y='tsne-two',
        palette=sns.color_palette("hls",10),
        data=df,
        legend="full",
        alpha=0.3
    )


def get_performance_metrics(speaker_df, outputs):
    agent_labels = np.array(speaker_df.Agent.tolist())
    caller_labels = np.array(speaker_df.Caller.tolist())



    agent_output_mask = agent_labels*outputs
    caller_output_mask = caller_labels*outputs
    #print(agent_output_mask)
    #print(caller_output_mask)

    agent_coverage = np.count_nonzero(agent_output_mask == 1)/np.count_nonzero(agent_output_mask != 0)
    caller_coverage = np.count_nonzero(caller_output_mask == 2)/np.count_nonzero(caller_output_mask != 0)

    agent_purity = np.count_nonzero(agent_output_mask == 1)/np.count_nonzero(outputs == 1)
    caller_purity = np.count_nonzero(caller_output_mask == 2)/np.count_nonzero(outputs == 2)

    return (agent_coverage+caller_coverage)/2, (agent_purity+caller_purity)/2

def merge_frames(outputs, frame_list, frame_labels):
    annotation = Annotation()
    seg_start = 0
    active_spk = 3
    #print(frame_labels)
    for i, frame in enumerate(frame_list):
        if frame_labels[i] != 0:
            try:
                annotation[Segment(start=float(frame[0]), end=float(frame[1]))] = outputs[i]
            except:
                print('{} is does not have an output but it has label {}'.format(frame, frame_labels[i]))
                print('i = {} and len = {}'.format(i, len(outputs)))
    annotation = annotation.support()
    #print(annotation.get_timeline())
    return annotation


def get_der(cfg, rttm, output_annotations):
    metric = DiarizationErrorRate(skip_overlap=False, collar=cfg.audio.collar)
    groundtruth = load_rttm(rttm)[rttm[rttm.rfind('/')+1:rttm.find('.')]]
    der = metric(groundtruth, output_annotations, detailed=False)
    return der

seed_everything(42)

def get_track_embeddings(cfg):
    '''
    Write on manifest for all window-size/overlaps and embeds all segments once which for each track which saves
    considerable time
    :param cfg: DictConfig file, used with hydra
    :return: fileids --> all the embedded files, track_manifest--> entire track_manifest.json file read into list
    '''

    #GPU access
    cuda = 1 if torch.cuda.is_available() else 0

    #Load model, take a look at ExtractSpeakerEmbeddingsModel function I have changed the code to give each embedding
    #a unique name (test loop)!!!
    model = ExtractSpeakerEmbeddingsModel.from_pretrained(model_name='SpeakerNet_verification')

    #load track names
    if cfg.audio.num_target_tracks > -1:
        fileids = [line[:line.rfind('\n')] for line in open(cfg.audio.fileids)][0:cfg.audio.num_target_tracks]
        #audio_tracks = glob.glob(cfg.audio.target_path, recursive=True)[:cfg.audio.num_target_tracks]
    else:
        fileids = [line[:line.rfind('\n')] for line in open(cfg.audio.fileids)][0:cfg.audio.num_target_tracks]
        #audio_tracks = glob.glob(cfg.audio.target_path, recursive=True)
    print(fileids)

    #remove the track_manifest if it exists
    #probably better to do with os.remove but this is easier
    os.system('rm -f {}'.format(cfg.audio.track_manifest))

    #write test-config for extracting model embeddings

    test_config = OmegaConf.create(dict(
        manifest_filepath=cfg.audio.track_manifest,
        sample_rate=16000,
        labels=None,
        batch_size=16,
        shuffle=False,

        embedding_dir=cfg.audio.embedding_dir,
        num_workers=4
    ))

    for window_length in cfg.audio.window_length:
        for step_length in cfg.audio.step_length:
            for track in fileids:
                label_path = cfg.audio.label_path + track + '.labs'
                frame_list, frame_labels = label_frames(label_path=label_path
                                                      , window_size=window_length
                                                      , step_size=float(window_length * step_length))
                write_track_manifest(audio_path=cfg.audio.target_path + track + '.wav', frame_list=frame_list,
                                     manifest_file=cfg.audio.track_manifest, window_length=window_length,
                                     step_length=step_length)
    model.setup_test_data(test_config)
    trainer = pl.Trainer(gpus=cuda)
    trainer.test(model)
    track_manifest = [json.loads(line.replace('\n', '')) for line in
                      open(cfg.audio.track_manifest)]
    return fileids, track_manifest


def vbhmm(embeddings, transform, plda):
    pca = PCA(n_components=256)
    embeddings = pca.fit_transform(embeddings)

    kaldi_plda = read_plda(plda)
    plda_mu, plda_tr, plda_psi = kaldi_plda
    W = np.linalg.inv(plda_tr.T.dot(plda_tr))
    B = np.linalg.inv((plda_tr.T / plda_psi).dot(plda_tr))
    acvar, wccn = eigh(B, W)
    plda_psi = acvar[::-1]
    plda_tr = wccn.T[::-1]

    #print(len(embeddings))
    print(np.vstack(embeddings).shape)
    with h5py.File(transform, 'r') as f:
        mean1 = np.array(f['mean1'])
        mean2 = np.array(f['mean2'])
        lda = np.array(f['lda'])
        embeddings = l2_norm(lda.T.dot((l2_norm(embeddings-mean1)).transpose()).transpose() - mean2)

    scr_mx = cos_similarity(embeddings)
    thr, junk = twoGMMcalib_lin(scr_mx.ravel())
    #labels1st = AHC(scr_mx, thr + -0.015)
    kmean_cluster = KMeans(n_clusters=2, random_state=5)
    labels1st = kmean_cluster.fit_predict(X=embeddings)
    #print(labels1st)
    # kmeans_cluster = AgglomerativeClustering(n_clusters=2, affinity='cosine', linkage='average')
    #kmeans_cluster.fit_predict(X=track_embedding)
    #labels1st = AgglomerativeClustering(n_clusters=2, affinity='cosine', linkage='average').fit(embeddings).labels_
    #print(labels1st)

    # Smooth the hard labels obtained from AHC to soft assignments
    # of x-vectors to speakers
    qinit = np.zeros((len(labels1st), np.max(labels1st) + 1))
    qinit[range(len(labels1st)), labels1st] = 1.0
    qinit = softmax(qinit * 5.0, axis=1)
    fea = (embeddings - plda_mu).dot(plda_tr.T)[:, :128]
    # Use VB-HMM for x-vector clustering. Instead of i-vector extractor model, we use PLDA
    # => GMM with only 1 component, V derived accross-class covariance,
    # and iE is inverse within-class covariance (i.e. identity)
    sm = np.zeros(128)
    siE = np.ones(128)
    sV = np.sqrt(plda_psi[:128])
    q, sp, L = VB_diarization(
        fea, sm, np.diag(siE), np.diag(sV),
        pi=None, gamma=qinit, maxSpeakers=qinit.shape[1],
        maxIters=40, epsilon=1e-6,
        loopProb=0.99, Fa=0.3, Fb=17)
    labels1st = np.argsort(-q, axis=1)[:, 0]
    #print(labels1st)
    if q.shape[1] > 1:
        labels2nd = np.argsort(-q, axis=1)[:, 1]
    #print(labels2nd)
    return labels2nd


@hydra.main(config_path='SpeakerNet_Diarisation_EMRAI.yaml')
def main(cfg: DictConfig) -> None:
    # remove the track_manifest if it exists
    # probably better to do with os.remove but this is easier
    os.system('rm -f {}'.format(cfg.audio.track_manifest))
    fileids, track_manifest = get_track_embeddings(cfg)

    with open(os.path.join(cfg.audio.embedding_dir,'embeddings/track_manifest_embeddings.pkl'), 'rb') as f:
        data = pickle.load(f).items()
        all_track_embeddings = [emb for _, emb in data]

    for window_length in cfg.audio.window_length:
        for step_length in cfg.audio.step_length:
            for track in fileids:
                label_path = cfg.audio.label_path+track+'.labs'
                rttm = cfg.audio.rttm_path+track+'.rttm'
                frame_list, frame_labels = label_frames(label_path=label_path,
                                                          window_size=window_length,
                                                          step_size=float(window_length * step_length))
                indices = [track_manifest.index(item) for item in track_manifest if item['audio_filepath'] == cfg.audio.target_path+track+'.wav' and item["duration"] == window_length and item["step_length"] == step_length]
                embedddings = all_track_embeddings[min(indices):max(indices)+1]
                #if cfg.audio.resegmentation:
                #    cluster_outputs = vbhmm(embeddings=embedddings, transform=cfg.audio.plda_backend, plda=cfg.audio.plda_file)
                #    annotation = merge_frames(outputs=cluster_outputs, frame_list=frame_list, frame_labels=frame_labels)
                #else:
                cluster_outputs = cluster_embeddings(track_embedding=embedddings)
                print(len(cluster_outputs))
                annotation = merge_frames(outputs=cluster_outputs, frame_list=frame_list, frame_labels=frame_labels)
                os.system('mkdir -p {}'.format(os.path.join(cfg.audio.out_rttm_path,'window-length-{}'.format(window_length), 'step-length-{}'.format(step_length))))
                with open(os.path.join(os.path.join(cfg.audio.out_rttm_path,'window-length-{}'.format(window_length), 'step-length-{}'.format(step_length),'{}.rttm'.format(track))),'w') as file:
                    annotation.write_rttm(file)




if __name__ == '__main__':
    main()
