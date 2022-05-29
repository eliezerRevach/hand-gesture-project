import math

import numpy as np
import matplotlib.pyplot as plt
path="SAME_AS_PROJECT_CSV" # input
Class="1"
File="20.csv"
mean_size=30
var_size=2
WINDOW_SIZE=50
do_mean=True
def create_plot(arr,color='red'):
  x = np.arange(0, len(arr))
  y = arr
  plt.title("Line graph")
  plt.xlabel("X axis")
  plt.ylabel("Y axis")
  plt.plot(x, y, color=color)

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


def RMS(Array2d_result):
  return np.sqrt(np.mean(Array2d_result ** 2))
  pass

def rmsValue(array):
  n = len(array)
  squre = 0.0
  root = 0.0
  mean = 0.0

  # calculating Squre
  for i in range(0, n):
    squre += (array[i] ** 2)
  # Calculating Mean
  mean = (squre / (float)(n))
  # Calculating Root
  root = math.sqrt(mean)
  return root

def mean_array(arr):
    re_Arr = np.array([0.] * (len(arr)))
    for i in range(len(arr) - (mean_size - 1)):
      re_Arr[i] = np.mean(arr[i:i + mean_size])

    return re_Arr
    pass


def var_array(arr):
  re_Arr = np.array([0.] * (len(arr)))
  for i in range(len(arr) - (var_size - 1)):
    re_Arr[i] = np.var(arr[i:i + var_size])
  return re_Arr
  pass


def twoD_mean_array(Array2d_result):
  re_array = Array2d_result
  for i in range(len(Array2d_result)):
    arr = mean_array(Array2d_result[i])
    re_array = np.vstack([re_array, arr])
  return re_array
  pass
def FFT(Array2d_result):
  re_array = Array2d_result
  for i in range(len(Array2d_result)):
    signal = Array2d_result[i]
    n = signal.size
    timestep = 1
    arr = np.fft.fftfreq(n, d=timestep)

    re_array = np.vstack([re_array, arr])
  return re_array
  pass


def twoD_var_array(Array2d_result):
  re_array = Array2d_result
  for i in range(len(Array2d_result)):
    arr = var_array(Array2d_result[i])
    re_array = np.vstack([re_array, arr])
  if do_mean:
    return re_array[:int(len(re_array) - (len(Array2d_result) / 2))]
  return re_array
  pass

def rolling_rms(x, N):
  xc = np.cumsum(abs(x) ** 2);
  # print(xc)
  return np.sqrt((xc[N:] - xc[:-N]) / N)
Array2d_result = np.genfromtxt(path + "/" + Class + "/" + File, delimiter=',')



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

mean_arr=[]
std_arr=[]
for i in range(len(Array2d_result)):
  mean,std=norm_histogram(Array2d_result[i])
  mean_arr.append(mean)
  std_arr.append(std)
print(mean_arr,std_arr)

from scipy.stats import norm


xmin, xmax = plt.xlim()
x = np.linspace(-300, 1000, 100)
p = norm.pdf(x, mean_arr[0], std_arr[0])
create_plot(np.absolute(Array2d_result[0]/20000))
arr=rolling_rms(Array2d_result[0],WINDOW_SIZE)

plt.plot(x, p, 'k', linewidth=3)
plt.axvline(mean_arr[0],color='green', linewidth=3)
plt.axvline(mean_arr[0]+2*std_arr[0], linewidth=3)
plt.axvline(mean_arr[0]-1.5*std_arr[0], linewidth=3)
title = "Fit results: mu = %.2f,  std = %.2f" % (mean_arr[0], std_arr[0])
plt.title(title)
plt.show()

mean,std=norm_histogram(arr)
p = norm.pdf(x, mean,std)
plt.plot(x, p, 'k', linewidth=3)
title = "Fit results: mu = %.2f,  std = %.2f" % (mean,std)
plt.axvline(mean,color='green', linewidth=3)
plt.axvline(mean+2*std, linewidth=3)
plt.axvline(mean-1.5*std, linewidth=3)

create_plot(arr/5000,color='blue')
plt.title(title)
plt.show()




xmin, xmax = plt.xlim()
x = np.linspace(-300, 1000, 100)
p = norm.pdf(x, mean_arr[1], std_arr[1])
create_plot(np.absolute(Array2d_result[1]/20000))
arr=rolling_rms(Array2d_result[1],WINDOW_SIZE)

plt.plot(x, p, 'k', linewidth=3)
plt.axvline(mean_arr[1],color='green', linewidth=3)
plt.axvline(mean_arr[1]+2*std_arr[1], linewidth=3)
plt.axvline(mean_arr[1]-1.5*std_arr[1], linewidth=3)
title = "Fit results: mu = %.2f,  std = %.2f" % (mean_arr[1], std_arr[1])
plt.title(title)
plt.show()

mean,std=norm_histogram(arr)
p = norm.pdf(x, mean,std)
plt.plot(x, p, 'k', linewidth=3)
title = "Fit results: mu = %.2f,  std = %.2f" % (mean,std)
plt.axvline(mean,color='green', linewidth=3)
plt.axvline(mean+2*std, linewidth=3)
plt.axvline(mean-1.5*std, linewidth=3)

create_plot(arr/5000,color='blue')
plt.title(title)
plt.show()





Array2d_result=np.absolute(Array2d_result)
WINDOW_SIZE=50#MS
arr=rolling_rms(Array2d_result[0],WINDOW_SIZE)
#MVC=MEAN OF MAX POWER
MVC=25
# print(arr)
# plotArray(arr/25)
