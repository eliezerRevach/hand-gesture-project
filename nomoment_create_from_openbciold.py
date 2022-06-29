#-----------------------------------------------imports------------------------------------------------#
import os

import numpy as np
from pylsl import StreamInlet, resolve_stream
from witmotion import IMU

#------------------------------------------------------------------------------------------------------#
# params:
sample_size = 512
output_path = "nomove_withimu/paper/"
channel_from = 2
channel_to = 4
limit=200
imu = IMU(path="COM8")# usb port of IMU
# -----------------------#

#--------1d array to 2d array-----------#



def to2DArr(data,channel_from,channel_to,resizeTo,reshape_to=4):
    A=np.reshape(data, (-1, reshape_to)).T
    A=np.pad(A, ((0, 0), (0, resizeTo-len(A[0]))))

    return A[channel_from:channel_to]
    pass
#-------------------------------------#

def LastNumberInFolder(path):# check witch file name number is last, returning is value
    arr = os.listdir(path)
    numbers=[]
    for item in arr:
        for subitem in item.split():
            if (subitem[:-4].isdigit()):
                numbers.append(subitem[:-4])
    if not numbers:
        return -1
    numbers=np.array(numbers)
    return max(numbers.astype(int))
    pass
def imu_current_sample():# get acceletraion and velocity as one sample , return zeros if not found

    a = imu.get_acceleration()
    b = imu.get_angular_velocity()
    if a and b:
        return np.append(a,b)
    else:
        return np.array([0,0,0,0,0,0])


def openbci_stream(sample_size,output, channel_from, channel_to, resizeTo):
    print("Looking for an EMG stream...")
    streams = resolve_stream('type', 'EMG')
    inlet = StreamInlet(streams[0])
    print("EMG stream found!")

    counter=LastNumberInFolder(output)
    for j in range(limit):
        counter += 1
        data_emg=np.array([])
        data_imu = np.array([])
        for i in range(sample_size):
            sample, timestamp = inlet.pull_sample()  # get EMG data sample and its timestamp

            data_emg=np.append(data_emg,sample)
            data_imu=np.append(data_imu,imu_current_sample())

        to_send_emg=to2DArr(data_emg, channel_from, channel_to, resizeTo,reshape_to=4)
        to_send_imu = to2DArr(data_imu, 0, 6, resizeTo,reshape_to=6)
        np.savetxt(output + str(counter) + '.csv', np.vstack((to_send_emg,to_send_imu)), delimiter=",")
        print("samples done:",j)
    print("finished")

def main():
    print("press enter to start")
    x=input()
    print("starting")
    isExist = os.path.exists(output_path)
    if not isExist:
        os.makedirs(output_path)
    openbci_stream(sample_size,output_path, channel_from, channel_to, sample_size)

main()
