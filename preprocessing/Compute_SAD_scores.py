import numpy as np
import webrtcvad
import glob
import librosa
from math import ceil
from sklearn.metrics import f1_score
from scipy.io import wavfile
import struct
import itertools
import torch
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Segment
import sys

def get_aspire_sad_results(file_id, segment_file, duration):
    results = [(float(segment.split()[2]), float(segment.split()[3])) for segment in open(segment_file) if segment.split()[1] == file_id]
    #print(results)
    labels = []
    #0's to first segment
    start = int(round(results[0][0],2)/0.01)
    labels.extend([0]*start)
    for i, segment in enumerate(results):
        speech_dur = int(round(segment[1] - segment[0],2)/0.01)
        labels.extend([1]*speech_dur)
        if segment != results[-1]:
            silence_dur = int(round(results[i+1][0]-segment[1],2)/0.01)
            labels.extend([0]*silence_dur)
        else:
            silence_dur = int(round(duration-segment[1],2)/0.01)
            labels.extend([0]*silence_dur)

    return labels


def run_pyannote_sad(file_id, audio_dir):
    pyannote_sad = torch.hub.load('pyannote/pyannote-audio', 'sad_dihard')
    test_file = {'uri': file_id,
                 'audio':audio_dir+'/'+file_id+'.wav'}
    sad_scores = pyannote_sad(test_file)
    binarize = Binarize(offset=0.52, onset=0.52, log_scale=True,
                        min_duration_off=0.1, min_duration_on=0.1)
    speech_labels = binarize.apply(sad_scores, dimension=1)

    labels = []
    start = int(round(speech_labels[0].start,2)/0.01)
    labels.extend([0]*start)
    for i, segment in enumerate(speech_labels):
        speech_dur = int(round(segment.end - segment.start,2)/0.01)
        labels.extend([1]*speech_dur)
        if segment != speech_labels[-1]:
            silence_dur = int(round(speech_labels[i+1].start - segment.end,2)/0.01)
            labels.extend([0] * silence_dur)
    return labels





def run_webrtc(wave_file):
    vad = webrtcvad.Vad()
    vad.set_mode(2)
    sample_rate, audio = wavfile.read(wave_file)
    print(sample_rate)
    raw_samples = struct.pack("%dh" % len(audio), *audio)

    window_duration = 0.03
    samples_per_window = int(window_duration * sample_rate)# + 0.5)
    bytes_per_sample = 2
    segments = []
    for start in np.arange(0, len(audio), samples_per_window):
        stop = min(start + samples_per_window, len(audio))
        try:
            is_speech = vad.is_speech(raw_samples[start * bytes_per_sample:stop * bytes_per_sample],
                                      sample_rate=sample_rate)
            segments.append(1 if is_speech else 0)
        except:
            segments.append(0)
    segments = list(itertools.chain.from_iterable(itertools.repeat(x, 3) for x in segments))
    return segments

def main():
    f1_SAIGEN = []
    f1_webrtc = []
    f1_pyannote = []
    lab_files = sys.argv[1]
    file_ids = sys.argv[2]
    aspire_labs = sys.argv[3]
    audio_dir = sys.argv[4]
    print('Reading lab files at {}'.format(lab_files))
    print('Reading fileids at {}'.format(file_ids))
    print('Reading aspire labs at {}'.format(aspire_labs))
    print('Reading audio files at {}'.format(audio_dir))
    file_ids = [line.split()[0] for line in open(file_ids)]
    lab_files = [[1 if int(label)>0 else 0 for label in open(lab_files+'/'+file_id+'.labs')] for file_id in file_ids]
    for i in range(len(file_ids)):
        Aspire_SAD = get_aspire_sad_results(file_id=file_ids[i], segment_file=aspire_labs, duration=len(lab_files[i])*0.01)
        print(lab_files[i])
        print(len(Aspire_SAD))
        if len(lab_files[i]) > len(Aspire_SAD):
            Aspire_SAD.extend([0] * (len(lab_files[i])-len(Aspire_SAD)))
        elif len(Aspire_SAD) > len(lab_files[i]):
            lab_files.extend([0]*(len(Aspire_SAD)-len(lab_files)))
        f1_SAIGEN.append (f1_score(lab_files[i], Aspire_SAD))
    #print(f1_SAIGEN)
    print('f1-score for SAIGEN labels {}'.format(sum(f1_SAIGEN)/len(f1_SAIGEN)))
    for i in range(len(file_ids)):
        print(len(lab_files[i]))
        WebRtcVad = run_webrtc(audio_dir + '/' + file_ids[i] + '.wav')
        print(len(WebRtcVad))
        if len(lab_files[i]) > len(WebRtcVad):
            WebRtcVad.extend([0] * (len(lab_files[i])-len(WebRtcVad)))
        elif len(WebRtcVad) > len(lab_files[i]):
            lab_files[i].extend([0]*(len(WebRtcVad) - len(lab_files[i])))
        f1_webrtc.append (f1_score(lab_files[i], WebRtcVad))
    print('f1-score for webrtcvad {}'.format(sum(f1_webrtc)/len(f1_webrtc)))
    for i in range(len(file_ids)):
        pyannote_sad_labels = run_pyannote_sad(file_ids[i], audio_dir)
        if len(lab_files[i]) > len(pyannote_sad_labels):
            pyannote_sad_labels.extend([0]*(len(lab_files[i])-len(pyannote_sad_labels)))
        elif len(pyannote_sad_labels) > len(lab_files[i]):
            lab_files[i].extend([0]*(len(pyannote_sad_labels)-len(lab_files[i])))
        f1_pyannote.append(f1_score(lab_files[i], pyannote_sad_labels))
    print('pyannote labels {}'.format(sum(f1_pyannote)/len(f1_pyannote)))

if __name__ == '__main__':
    main()


