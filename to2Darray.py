#---------parameters---------#
constSize=False
#-----------#
resizeTo=512*3
folder="test_all/"# input
output_folder="test_all_2d"
channel_from=2
channel_to=4
raw_exist=0
#-----------#


#--------transform to an image-----------#



def to2DArr(data):

    A=np.reshape(data, (-1, 4)).T

    A=np.pad(A, ((0, 0), (0, resizeTo-len(A[0]))))
    if constSize:
        A = A.resize((resizeTo,resizeTo))
    return A[channel_from:channel_to]
    pass

#----------------------------------------#


#-----------loading data-----------------# use openData(path)
import os
import numpy as np
def load_file(file_path):
    txt=open(file_path).read()
    alist=[]
    for line in txt.splitlines():
        if '...' not in line:
            alist.append([float(i) for i in line.split(',')])
    return alist

def openData(dir_name):

    folders = os.listdir(dir_name)
    data_List=[np.array([])]*(len(folders)-raw_exist)
    type_Counter=0

    for folder_name in folders:


        if folder_name!="raw" and folder_name!="with_last":
            folder=os.listdir(dir_name+"/"+folder_name)
            for file_name in folder:
                file_path=dir_name+"/"+folder_name+"/"+file_name
                data=load_file(file_path)
                print("load_file",type_Counter,"from",folder_name)
                if len(data_List[type_Counter])==0 :
                    data_List[type_Counter]=data
                else:
                    data_List[type_Counter]=np.concatenate((data_List[type_Counter],data))
        type_Counter+=1
    return data_List
    pass
#----------------------------------------#



data=openData(folder)#3d array, [hand gesture type(0,1,2),sample arrays]
for i in range(len(data)):
    print(i)
    for j in range(len(data[i])):
        print(j)
        path=output_folder+"/"+str(i)+"/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        text=to2DArr(data[i][j])
        np.savetxt(path+str(j)+'.csv', text, delimiter=",")

        pass
