from statistics import mean, stdev
from sys import argv

rttm_file = argv[1]

res = [float(line.split()[1]) for line in open(rttm_file)]
print(mean(res))
print(stdev(res))
