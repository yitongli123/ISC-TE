import csv
import os 
file_name = os.listdir('./example_re/images/airplane/')
print(file_name)
data = []
num = 1
for file in file_name:
    temp = []
    temp.append(file[:-4])
   
    if num>385:
        temp.append('0')
    else:
        temp.append('1')
    num = num+1
    data.append(temp)
headers = ['ImageID','fold']
print(data)
with open('./example_re/labels/airplane/train.csv','w')as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(data)
