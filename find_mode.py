
from collections import defaultdict
from os import listdir



path = "words/"
file_names = listdir(path)
for file_name in file_names:
    file_path =  path + file_name
    L=[]
    with open(file_path, 'r') as f:
        lines = f.readlines()
        align = [(int(float(y[3]))-int(float(y[2]))) for y in [x.strip().split(" ") for x in lines]]

    for k in align:
        L.append(k)
    mx = max(L)
    d = defaultdict(int)
    for i in L:
        d[i] += 1
    result = max(d.iteritems(), key=lambda x: x[1])
    st = file_name + " " + str(result[0])+" "+str(mx)
    print(result)
    with open("find_mode.txt", "a") as myfile:
        myfile.write(st)
        myfile.write('\n')





