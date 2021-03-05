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

<<<<<<< HEAD
#import sys
=======
import sys
>>>>>>> c890048d7f13643095d9f34287c876b9bd9a482f


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
            annotation[Segment(start=float(frame[0]), end=float(frame[1]))] = outputs[i]
    annotation.support(collar=0)
    #print(annotation.get_timeline())
    return annotation


def get_der(cfg, rttm, output_annotations):
    metric = DiarizationErrorRate(skip_overlap=False, collar=cfg.audio.collar)
    groundtruth = load_rttm(rttm)[rttm[rttm.rfind('/')+1:rttm.find('.')]]
    der = metric(groundtruth, output_annotations, detailed=False)
    return der

seed_everything(42)

<<<<<<< HEAD
def get_track_embeddings(cfg):
    '''
    Write on manifest for all window-size/overlaps and embeds all segments once which for each track which saves
    considerable time
    :param cfg: DictConfig file, used with hydra
    :return: fileids --> all the embedded files, track_manifest--> entire track_manifest.json file read into list
    '''
    # GPU access
    cuda = 1 if torch.cuda.is_available() else 0

    # Load model, take a look at ExtractSpeakerEmbeddingsModel function I have changed the code to give each embedding
    # a unique name (test loop)!!!
    model = ExtractSpeakerEmbeddingsModel.from_pretrained(model_name='SpeakerNet_verification')

    # load track names
    if cfg.audio.num_target_tracks == -1:
        fileids = [line[:line.rfind('\n')] for line in open(cfg.audio.fileids)]
    else:
        fileids = [line[:line.rfind('\n')] for line in open(cfg.audio.fileids)][0:cfg.audio.num_target_tracks]

    # write test-config for extracting model embeddings
=======
@hydra.main(config_path='SpeakerNet_Diarisation_EMRAI.yaml')
def main(cfg: DictConfig) -> None:

    #GPU access
    cuda = 1 if torch.cuda.is_available() else 0

    #Load model, take a look at ExtractSpeakerEmbeddingsModel function I have changed the code to give each embedding
    #a unique name (test loop)!!!
    model = ExtractSpeakerEmbeddingsModel.from_pretrained(model_name='SpeakerNet_verification')

    #load track names
    if cfg.audio.num_target_tracks > -1:
        fileids = [line for line in cfg.audio.fileids]
        #audio_tracks = glob.glob(cfg.audio.target_path, recursive=True)[:cfg.audio.num_target_tracks]
    else:
        fileids = [line for line in cfg.audio.fileids][0:cfg.audio.num_target_tracks]
        #audio_tracks = glob.glob(cfg.audio.target_path, recursive=True)
    print(fileids)

    #remove the track_manifest if it exists
    #probably better to do with os.remove but this is easier
    os.system('rm -f {}'.format(cfg.audio.track_manifest))

    #write test-config for extracting model embeddings
>>>>>>> c890048d7f13643095d9f34287c876b9bd9a482f
    test_config = OmegaConf.create(dict(
        manifest_filepath=cfg.audio.track_manifest,
        sample_rate=16000,
        labels=None,
        batch_size=16,
        shuffle=False,
<<<<<<< HEAD
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
                cluster_outputs = cluster_embeddings(track_embedding=embedddings)

                annotation = merge_frames(outputs=cluster_outputs, frame_list=frame_list, frame_labels=frame_labels)
                os.system('mkdir -p {}'.format(os.path.join(cfg.audio.out_rttm_path,'window-length-{}'.format(window_length), 'step-length-{}'.format(step_length))))
                with open(os.path.join(os.path.join(cfg.audio.out_rttm_path,'window-length-{}'.format(window_length), 'step-length-{}'.format(step_length),'{}.rttm'.format(track))),'w') as file:
                    annotation.write_rttm(file)

                #der = get_der(cfg=cfg, rttm=rttm, output_annotations=annotation)
                #print('der for {} is {}'.format(track, der))
=======
        embedding_dir='./',
        num_workers = 4
    ))



    #for window_length in cfg.audio.window_length:
    #    for step_length in cfg.audio.step_length:
    #        for track in audio_tracks:
    #            agent = track[track.find('-') + 1:track.find('.')]
    #            agent_samples = glob.glob(cfg.audio.verification_path + agent + '.wav', recursive=True)
    #            rttm = glob.glob(cfg.audio.rttm_path + track[track.rfind('/') + 1:track.rfind('.')] + '.rttm',
    #                             recursive=False)[0]
    #            #print(agent_samples)
    #            if len(agent_samples) > 0:
    #                label_path = track[track.rfind('/')+1:track.find('.wav')]+'.labs'
    #                frame_list, speaker_df = label_frames(label_path=os.path.join(cfg.audio.label_path, label_path),
    #                                                  window_size=window_length,
    #                                                  step_size=float(window_length*step_length))
    #                write_track_manifest(audio_path=track, frame_list=frame_list, manifest_file='track_manifest.json', window_length=window_length, step_length=step_length)
    #model.setup_test_data(test_config)
    #trainer = pl.Trainer(gpus=cuda)
    #trainer.test(model)
    #track_manifest = [json.loads(line.replace('\n', '')) for line in
    #                  open(os.path.join(os.getcwd(), 'manifest_files', 'track_manifest.json'))]
    #with open(os.path.join(os.getcwd(),'embeddings','track_manifest_embeddings.pkl'), 'rb') as f:
    #    data = pickle.load(f).items()
    #    all_track_embeddings = [emb for _, emb in data]
    #for window_length in cfg.audio.window_length:
    #    for step_length in cfg.audio.step_length:
    #        for track in audio_tracks:
    #            agent = track[track.find('-') + 1:track.find('.')]
    #            agent_samples = glob.glob(cfg.audio.verification_path + agent + '.wav', recursive=True)
    #            rttm = glob.glob(cfg.audio.rttm_path + track[track.rfind('/') + 1:track.rfind('.')] + '.rttm',
    #                             recursive=False)[0]
    #            # print(agent_samples)
    #            if len(agent_samples) > 0:
    #                label_path = track[track.rfind('/') + 1:track.find('.wav')] + '.labs'
    #                frame_list, speaker_df = label_frames(label_path=os.path.join(cfg.audio.label_path, label_path),
    #                                                      window_size=window_length,
    #                                                      step_size=float(window_length * step_length))
    #                indices = [track_manifest.index(item) for item in track_manifest if
    #                           item['audio_filepath'] == track and item["duration"] == window_length and item[
    #                               "step_length"] == step_length]
    #                print(indices)
    #                embedddings = all_track_embeddings[min(indices):max(indices)+1]
    #                cluster_outputs = cluster_embeddings(agent=agent, track=track, window_length=window_length, step_length=step_length, track_embedding=embedddings)
    #                #print(len(cluster_outputs))
    #                #print(speaker_df.describe())
    #                coverage, purity = get_performance_metrics(speaker_df, np.array(cluster_outputs))
    #                print("The results for {} -> Coverage {} / Purity {}".format(track, coverage, purity))
    #                annotation = merge_frames(outputs=cluster_outputs, frame_list=frame_list)
    #                der = get_der(cfg=cfg, rttm=rttm, output_annotations=annotation)
>>>>>>> c890048d7f13643095d9f34287c876b9bd9a482f
    #                print('THE DER IS {}'.format(der))
    #                der_log.write('{} \t {} \t {} \t {} \t {} \t {} \n'.format(track, window_length, step_length, coverage,
    #                                                                 purity, der))

    #der_log.close()


if __name__ == '__main__':
    main()
 #Write target-speaker manifest files and check to see that all audio files have matching target files
    #for track in audio_tracks:
    #    agent=track[track.find('-')+1:track.find('.')]
    #    agent_samples = glob.glob(cfg.audio.verification_path+agent+'.wav', recursive=True)
    #    if len(agent_samples) > 0:
    #        write_target_manifest(audio_path=agent_samples[0], length=cfg.audio.verification_length, manifest_file='target.json',agent=agent)
    #        # write_track_manifest(audio_path=track, frame_list=frame_list, manifest_file='track_manifest.json')
    #        #model.setup_test_data(write_target_manifest(audio_path=agent_samples[0], length=cfg.audio.verification_length, manifest_file='target.json',agent=agent))
    #        #trainer = pl.Trainer(gpus=cuda, accelerator=None)
    #        #trainer.test(model)
    #    else:
    #        warnings.warn('Verification audio for {} not found '.format(agent))
    #test_config = OmegaConf.create(dict(
    #    manifest_filepath = os.path.join(os.getcwd(), 'manifest_files', 'target.json'),
    #    sample_rate = 16000,
    #    labels = None,
    #    batch_size = 1,
    #    shuffle=False,
    #    embedding_dir='./'#os.path.join(os.getcwd(),'embeddings')
    #))
    #model.setup_test_data(test_config)
    #trainer = pl.Trainer(gpus=cuda)
    #trainer.test(model)