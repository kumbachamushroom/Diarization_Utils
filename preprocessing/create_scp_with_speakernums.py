import sys
import os

fileids = sys.argv[1]
out_dir = sys.argv[2]


#fileids = '/home/lucas/PycharmProjects/Data/EMRAI/dev_other/fileids'
#lab_dir = '/home/lucas/PycharmProjects/Data/EMRAI/dev_other/framelabs'
#out_dir = '/home/lucas/PycharmProjects/Diarization_Utils/preprocessing/EMRAI_VBx_ref_labs'

fileids = [line[:line.rfind('\n')] for line in open(fileids)]

with open(os.path.join(out_dir,'wav_spknum.scp'),'w') as f:
    for id in fileids:
        f.write('{} \t 2 \n'.format(id))

