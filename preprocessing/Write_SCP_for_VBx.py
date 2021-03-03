import os
import sys

file_ids = sys.argv[1]
segments = sys.argv[2]
out = sys.argv[3]


file_ids = [line.split()[0] for line in open(file_ids)]
segments = [[(line.split()[2], line.split()[3]) for line in open(segments) if line.split()[1] == id] for id in file_ids]

for id in file_ids:
    f = open(out+id+'.lab', "w")
    for seg in segments[file_ids.index(id)]:
        f.write(seg[0]+"\t"+seg[1]+"\t"+"sp"+"\r")
    f.close()
