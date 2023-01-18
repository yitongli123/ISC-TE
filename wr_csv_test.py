import csv
import os 
file_name = os.listdir('./example_re/test/airplane/')
print(file_name)
data = []
num = 1
for file in file_name:
    temp = []
    temp.append(file[:-4])
    temp.append('4')
    data.append(temp)
headers = ['ImageID','fold']
print(data)
with open('./example_re/labels/airplane/test.csv','w')as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(data)
