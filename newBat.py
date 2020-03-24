import array
import numpy
import os
import glob

path_file=glob.glob('pictures/EXE/*') #获取当前文件夹下个数
path_number=len(path_file)
print(path_number)
tol=0

for i in range(path_number):#批量生成objdump命令
    path_file_2=glob.glob(path_file[i]+'/*.exe')
    path_number_2=len(path_file_2)
    for j in range(path_number_2):
        filename=path_file_2[j]
        f = open('bash.txt','a')
        f.write('objdump -d '+'/tmp/'+filename+' > output'+str(tol)+'\n')
        tol=tol+1
        f.close()

xmls = glob.glob('bash.txt')
for one_xml in xmls:#把'\\'全部替换为'/'，所得命令可以直接在Linux命令行执行
    print(one_xml)
    f = open(one_xml, 'r+', encoding='utf-8')
    all_the_lines = f.readlines()
    f.seek(0)
    f.truncate()
    for line in all_the_lines:
        line = line.replace('\\', '/')
        f.write(line)
    f.close()

