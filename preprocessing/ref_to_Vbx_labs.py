import os
import sys

#Convert reference labels to .scp format for VBx diarisation
# Ref labels are 10ms labels, either 0 for non speech or 1 and 2 for speakers
# .scp file is <start> <end> sp and saved as file-id.lab
#Usage: python ref_to_Vbx_labs.py <file-ids> <lab-in> <out-dir>

#fileids = sys.argv[1]
#lab_dir = sys.argv[2]
#out_dir = sys.argv[3]

fileids = '/home/lucas/PycharmProjects/Data/EMRAI/dev_other/fileids'
lab_dir = '/home/lucas/PycharmProjects/Data/EMRAI/dev_other/framelabs'
out_dir = '/home/lucas/PycharmProjects/Diarization_Utils/preprocessing/EMRAI_VBx_ref_labs'

fileids = [line[:line.rfind('\n')] for line in open(fileids)]



for id in fileids:
    labs = [1 if int(label) > 0 else 0 for label in open(os.path.join(lab_dir,'{}.labs'.format(id)))]
    start = 0
    segments = []
    for i, lab in enumerate(labs):
        if lab > 0 and start == 0:
            start = (i+1)*0.01
        elif lab == 0 and start > 0:
            end = (i)*0.01
            segments.append((start,end))
            start = 0
    with open(os.path.join(out_dir, '{}.lab'.format(id)),'w') as file:
        for i, segment in enumerate(segments):
            if i < len(segments):
                file.write('{} \t {} \t sp \n'.format(round(segment[0],3), round(segment[1],3)))
            else:
                file.write('{} \t {} \t sp'.format(round(segment[0], 3), round(segment[1], 3)))
