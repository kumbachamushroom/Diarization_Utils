import os
import glob
from statistics import mean, stdev
from sys import argv
#Compute average DER for all window-sizes and step-size
#directories should be <window-sizes><step-sizes>

def compute_mean_std(res_file):
    results = [float(line.split()[1]) for line in open(res_file)]
    der_mean = mean(results)
    der_std = stdev(results)
    return der_mean, der_std

def main():
    dir = argv[1]
    with open(os.path.join(dir,'der_mean_std.txt'),'w') as f:
        window_lengths = glob.glob(dir+'/*', recursive=True)
        for window in window_lengths:
            step_lengths = glob.glob(window+'/*',recursive=True)
            for step in step_lengths:
                mean, std = compute_mean_std(res_file=step+'/der_results.txt')
                f.write('{}\t{}\t{}\n'.format(step, mean, std))

if __name__ == '__main__':
    main()