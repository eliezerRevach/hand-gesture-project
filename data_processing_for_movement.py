#params
import os
import random

import numpy as np
from matplotlib import pyplot as plt

normalize=True

do_rms=True
do_histogram=True

Class_output_name="movement"
max_power_file="free_records/for_mvc.txt"
path="rock_paper_scissor_380size_csv" # input
new_folder="test_movement"
deviation_right =0.25
deviation_left =0.5
reuse_sample=5
move_central_by_std= -0.75#move the center left or right depend on the std before doing random deviation

window_size=50# ms


add_zeros=True
#----#
complete_to_in_rms=512*2
complete_to_final=256
#----#



def updateFile(path, File,Array2d_result):
    if not os.path.exists(path):
        os.makedirs(path)
    np.savetxt(path+"/"+File, Array2d_result, delimiter=",")
    pass

def plotArray(arr):
  x=np.arange(0, len(arr))
  y=arr
  plt.title("Line graph")
  plt.xlabel("X axis")
  plt.ylabel("Y axis")
  plt.plot(x, y, color ="red")
  plt.show()
def into_graph(two_d_arr):
  for i in range(len(two_d_arr)):
    plotArray(two_d_arr[i])

def rolling_rms(x, N):
  xc = np.cumsum(abs(x) ** 2);
  return np.sqrt((xc[N:] - xc[:-N]) / N)

def toRMS(Array2d_result,MVC=None):
    re_array=Array2d_result
    array_len=len(Array2d_result)
    for i in range(len(Array2d_result)):
        arr=rolling_rms(Array2d_result[i],window_size)
        arr = np.pad(arr,(0, complete_to_in_rms - len(arr)))
        if normalize:
            arr=arr/MVC[i]
        re_array=np.vstack([re_array,arr])
    return re_array[array_len:]
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
  std=(std/(sum(arr)-1)) **0.5
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


def histogramTwoD(Array2d_result):
    arr=Array2d_result
    for i in range(len(Array2d_result)):
        hist, bin_edges =np.histogram(Array2d_result[i],bins = 10)
        plt.hist(Array2d_result[i], bins=complete_to_final)
        plt.title("histogram")
        plt.show()
    pass


folders = os.listdir(path)
MVC=None
if normalize:
    MVC = get_mvc(max_power_file)


def cut_sample(Array2d_result, mean_arr, deviation_right,deviation_left,complete_to_final):
    re_array=Array2d_result[:,0:complete_to_final]
    for i in range(len(Array2d_result)):
        central=random.randint(int(mean_arr[i]-deviation_left[i]),int(mean_arr[i]+deviation_right[i]))

        From=max(int(central-complete_to_final/2),0)
        To=min(int(central+complete_to_final/2),len(Array2d_result[i]))
        left_mis=max(int(complete_to_final/2)-central,0)
        right_mis=max((central+complete_to_final/2)-len(Array2d_result[i]),0)
        arr=Array2d_result[i][From:To]

        arr = np.pad(arr,(left_mis,right_mis))
        re_array=np.vstack([re_array,arr])
    return re_array[len(Array2d_result):]
    pass
    pass


for Class in folders:

    files = os.listdir(path + "/" + Class)
    for File in files:
        Array2d_result = np.genfromtxt(path + "/" + Class + "/" + File, delimiter=',')
        array_len = len(Array2d_result)
        if do_rms:
            Array2d_result = toRMS(Array2d_result, MVC)
        mean_arr, std_arr = twoD_normal_distribution(Array2d_result)
        mean_arr=np.add(mean_arr,np.array(std_arr)*move_central_by_std)
        for i in range(reuse_sample):
            Array2d_result_inloop=cut_sample(Array2d_result,mean_arr,deviation_right*np.array(std_arr),deviation_left*np.array(std_arr),complete_to_final)
            # if do_histogram:
            #     Array2d_result_inloop=histogramTwoD(Array2d_result_inloop)
            #     pass
            updateFile(new_folder+"/"+Class_output_name,str(Class)+"_"+File[:-4]+"_"+str(i)+File[-4:],Array2d_result_inloop)
