import os
import numpy as np

def rolling_rms(x, N):
  xc = np.cumsum(abs(x) ** 2);
  return np.sqrt((xc[N:] - xc[:-N]) / N)

def toRMS(Array2d_result,MVC=None,window_size=50,complete_to_final=200):
    #rms on 2d array, with a window size, if to do mvc normaliztion and output size
    re_array=Array2d_result[:,0:complete_to_final]
    array_len=len(Array2d_result)
    for i in range(len(Array2d_result)):
        arr=rolling_rms(Array2d_result[i],window_size)
        if complete_to_final<len(arr):
            arr=arr[0:complete_to_final]
        else:
            arr = np.pad(arr, (0, complete_to_final - len(arr)))
        if MVC:
            arr=arr/MVC[i]
        re_array=np.vstack([re_array,arr])
    return re_array[array_len:]
    pass


def var_by_frame(arr,frame_size):
    #creating a var array based on a given frame
    re_Arr = np.array([0.] * (len(arr)))
    for i in range(len(arr) - (frame_size - 1)):
        re_Arr[i] = np.var(arr[i:i + frame_size])
    tempArr = re_Arr[0:len(re_Arr) - frame_size]
    re_Arr = np.append([0.] * int(frame_size / 2), tempArr)
    re_Arr = np.append(re_Arr, [0.] * int(frame_size / 2))
    return re_Arr
    pass


def normal_distribution(arr, frame_size=50,var=False):# useing normal distribution on the data to get std and mean
  #getting an arr , acting as it was an histogram , and calculating mean and std, based on location * value
  arr=np.absolute(arr)
  if var:
    arr = var_by_frame(arr, frame_size)
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


def twoD_normal_distribution(Array2d_result,var_farme_size=50,var=False):# getting std and mean arrays' from each array , useing normal distribution on the data

    mean_arr = []
    std_arr = []
    for i in range(len(Array2d_result)):
        mean, std = normal_distribution(Array2d_result[i],frame_size=var_farme_size,var=var)
        mean_arr.append(mean)
        std_arr.append(std)
    return mean_arr,std_arr
    pass

def twoD_cut_by_std(Array2d_result, mean_arr, std_arr, std_from_multipli, std_to_multipli,complete_to):# cut a sample bettwen [mean-from*std] to [mean+to*std]
    re_array=Array2d_result[:,0:complete_to]
    print("mean:",mean_arr)
    print("std:",std_arr)
    """useing an std and mean for each channel , 
    putting the mean location of the sample as the middle in the output ,
    then based on std taking from each side , left and right from the mean , values 
    for the output, alot of cases as to be made (std bigger then input size or output size, mean close to an edge,etc..)
    """
    for i in range(len(Array2d_result)):
        mean_int=int(mean_arr[i])
        std_int=int(std_arr[i])
        std_left=int(std_int*std_from_multipli)
        std_right=int(std_int*std_to_multipli)

        from_mean_to_left_size= std_left if mean_int>std_left else mean_int
        from_mean_to_right=std_right if mean_int+std_right<len(Array2d_result[i]) else  len(Array2d_result[i])-mean_int

        arr=Array2d_result[i][mean_int-from_mean_to_left_size:mean_int+from_mean_to_right]

        half_of_complete_to=int(complete_to/2)
        add_zeros_to_left=half_of_complete_to-from_mean_to_left_size
        add_zeros_to_right=half_of_complete_to-from_mean_to_right

        if add_zeros_to_right<0 and add_zeros_to_left<0:
            arr=arr[from_mean_to_left_size-half_of_complete_to:from_mean_to_left_size+half_of_complete_to] # mean[i]= from mean to left

            add_zeros_to_left=0
            add_zeros_to_right=0

        elif add_zeros_to_right<0:
            arr=arr[0:from_mean_to_left_size+half_of_complete_to]
            add_zeros_to_right=0
        elif add_zeros_to_left<0:
            arr=arr[from_mean_to_left_size-half_of_complete_to:]
            add_zeros_to_left=0

        arr = np.pad(arr,(add_zeros_to_left,add_zeros_to_right))
        re_array=np.vstack([re_array,arr])
    return re_array[len(Array2d_result):]
    pass