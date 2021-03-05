'''
Script for computing DER/JER files
USAGE: python DIAR_Scoring.py <path/to/fileids> <path/to/reference/rttm/files> <path/to/hypothesis/rttm/files> <error-metric ('DER'/'JER')> <path/to/output>
'''

import os
import glob

from pyannote.database.util import load_rttm
from pyannote.core import Annotation
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
from sys import argv

def compute_DER(fileids, ref_path, hyp_path, out_path):
    fileids = [line[:line.rfind('\n')] for line in open(fileids)]
    os.system('mkdir -p {}'.format(out_path))
    with open(os.path.join(out_path, 'der_results.txt'), 'w') as file:
        for id in fileids:
            try:
                reference = load_rttm(os.path.join(ref_path,'{}.rttm'.format(id)))
                reference = next(iter(reference.values()))

                hypotheses = load_rttm(os.path.join(hyp_path,'{}.rttm'.format(id)))
                hypotheses = next(iter(hypotheses.values()))

                metric = DiarizationErrorRate(skip_overlap=True, collar=0.250)
                der = metric(reference, hypotheses, detailed=False)
                file.write('{} \t {} \n'.format(id, der))
            except:
                None

    return None

def main():
    print(argv)
    compute_DER(argv[1], argv[2], argv[3], argv[5])

if __name__ == '__main__':
    main()




