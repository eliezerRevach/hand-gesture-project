#params
import os

do_rms=True
normalize=True
do_normal_distribution=True
max_power_file="free_records/for_mvc.txt"
path="rock_paper_scissor_380size_csv" # input
new_folder="rock_paper_scissor_380size_RMS_CSV"
mean_size=10
var_size=10
window_size=50# ms
std_from_multipli=1.5#mean - std*multipli... -> start of sample
std_to_multipli=2#mean + std*multipli... -> end of sample

add_zeros=True
#----#
complete_to=512*3
#----#

import numpy as np

def rolling_rms(x, N):
  xc = np.cumsum(abs(x) ** 2);
  return np.sqrt((xc[N:] - xc[:-N]) / N)

def toRMS(Array2d_result,MVC=None):
    re_array=Array2d_result
    for i in range(len(Array2d_result)):
        arr=rolling_rms(Array2d_result[i],window_size)
        arr = np.pad(arr,(0, complete_to - len(arr)))
        if normalize:
            arr=arr/MVC[i]
        re_array=np.vstack([re_array,arr])
    return re_array[Array2d_result:]
    pass
def updateFile(path, File,Array2d_result):
    if not os.path.exists(path):
        os.makedirs(path)
    np.savetxt(path+"/"+File, Array2d_result, delimiter=",")
    pass



def load_file(file_path):
    txt = open(file_path).read()
    alist = []
    for line in txt.splitlines():
        if '...' not in line:
            alist.append([float(i) for i in line.split(',')])
    return alist

def get_mvc(file_path):
    return np.mean(np.abs(load_file(file_path)),axis=0)
    pass
def norm_histogram(arr):
  arr=np.absolute(arr)
  mean=0
  std=0
  for i in range(len(arr)):
    mean+=arr[i]*i
  mean=mean/sum(arr)
  for i in range(len(arr)):
    std+=arr[i]*((i-mean)**2)
  print(std)
  std=(std/(sum(arr)-1)) **0.5
  print(std)
  return mean,std
  pass


def twoD_normal_distribution(Array2d_result):
    mean_arr = []
    std_arr = []
    for i in range(len(Array2d_result)):
        mean, std = norm_histogram(Array2d_result[i])
        mean_arr.append(mean)
        std_arr.append(std)
    return mean_arr,std_arr
    pass


def twoD_cut_by_std(Array2d_result, mean_arr, std_arr, std_from_multipli, std_to_multipli):
    re_array=Array2d_result
    for i in range(len(Array2d_result)):
        arr=Array2d_result[i][int(max(mean_arr[i]-std_arr[i]*std_from_multipli,0)):int(min(mean_arr[i]+std_arr[i]*std_to_multipli,complete_to))]
        left_mis=int(max(mean_arr[i]-std_arr[i]*std_from_multipli,0))
        right_mis=complete_to-int(min(mean_arr[i]+std_arr[i]*std_to_multipli,complete_to))
        arr = np.pad(arr,(left_mis,right_mis))
        re_array=np.vstack([re_array,arr])
    return re_array[len(Array2d_result):]
    pass

folders = os.listdir(path)
MVC=None
if normalize:
    MVC = get_mvc(max_power_file)

for Class in folders:

    files = os.listdir(path + "/" + Class)
    for File in files:
        Array2d_result = np.genfromtxt(path + "/" + Class + "/" + File, delimiter=',')
        array_len=len(Array2d_result)
        if do_normal_distribution :
            mean_arr,std_arr=twoD_normal_distribution(Array2d_result)
            Array2d_result=twoD_cut_by_std(Array2d_result,mean_arr,std_arr,std_from_multipli,std_to_multipli)
            pass

        if do_rms:
            Array2d_result = toRMS(Array2d_result,MVC)

        updateFile(new_folder+"/"+Class,File,Array2d_result)