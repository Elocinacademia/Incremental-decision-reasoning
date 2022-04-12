#Straightlining check

import csv
from collections import Counter
import itertools

def stringToList(string):
    listRes = list(string.split(","))
    return listRes


f = open('../../Data/new_file.csv')
reader = csv.reader(f)






def find_max(a):
    return max([len(list(v)) for k,v in itertools.groupby(a)])


new_file = []
counter = 0
for index, row in enumerate(reader):
    count = []
    newStr = ""
    for num, item in enumerate(row):
        item = item.strip("[]")
        new_item = stringToList(item)
        # print(new_item)
        acceptable = new_item[-1].strip("' '")
        count.append(acceptable)
    # print(count)
    max_count = find_max(count)
    
    if max_count > 32:
        counter +=1
        print(max_count, index+1)
    else:
        new_file.append(row)
print(counter)


file = '../../Data/final_data.csv'
with open(file,"w") as csv_file:
    writer=csv.writer(csv_file)
    for key,value in enumerate(new_file):
        # import pdb; pdb.set_trace() 
        writer.writerow(value)





       
        
