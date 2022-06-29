import os
import numpy as np
r"""
*when there's non movements we just procces the data noramly like the others,
cut by the needed frames

*like the others , MVC is an option that can be used by adding the path to the record file 
"""



window_size=50
complete_to_final=450
normalize=True
max_power_file="free_records/for_mvc.txt"
path="nomove_withimu" # input
procces_channel_up_to=2
new_folder="nomove3_p/"
Class_output_name="0/"
folders = os.listdir(path)
MVC=None

def load_file(file_path):# load file and check if sample have no issues
    txt = open(file_path).read()
    alist = []
    for line in txt.splitlines():
        if '...' not in line:
            alist.append([float(i) for i in line.split(',')])
    return alist

def get_mvc(file_path):# get mean of a entire data file, used on the max power
    return np.mean(np.abs(load_file(file_path)),axis=0)
    pass




def updateFile(path, File,Array2d_result):# save file and create path if needed
    if not os.path.exists(path):
        os.makedirs(path)
    np.savetxt(path+"/"+File, Array2d_result, delimiter=",")
    pass



def rolling_rms(x, N):# doing rolling rms with a given window
  xc = np.cumsum(abs(x) ** 2);
  return np.sqrt((xc[N:] - xc[:-N]) / N)

def toRMS(Array2d_result,MVC=None):#doing rolling rms on a 2d array , include the mvc
    re_array=Array2d_result[:,0:complete_to_final]# remove later,  only to have an array to work on
    array_len=len(Array2d_result)
    for i in range(len(Array2d_result)):
        if i<procces_channel_up_to:
            arr=rolling_rms(Array2d_result[i],window_size)
            arr=arr[0:complete_to_final]
            if normalize:
                arr=arr/MVC[i]
        else:
            arr=Array2d_result[i]
            arr=arr[0:complete_to_final]
        re_array=np.vstack([re_array,arr])
    return re_array[array_len:]
    pass




if normalize:
    MVC = get_mvc(max_power_file)
counter=0
for Class in folders:
    print(Class)
    files = os.listdir(path + "/" + Class)
    for File in files:
        Array2d_result = np.genfromtxt(path + "/" + Class + "/" + File, delimiter=',')
        array_len = len(Array2d_result)
        Array2d_result = toRMS(Array2d_result, MVC)
        updateFile(new_folder+"/"+Class_output_name,str(counter)+File[-4:],Array2d_result)
        counter+=1