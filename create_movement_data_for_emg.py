import os
import pickle
import sys
import time

import asyncio

import winsound
from websockets import connect
import threading
import numpy as np
import json
from pylsl import StreamInlet, resolve_stream
from witmotion import IMU



LM=50000#lest to remmber with imu and emg
filename='model_movement.sav'# model movement
imu = IMU(path="COM8")# usb port of IMU
Arr_emg=np.array([[0.]*5]*LM)# created stack to remmber LM Last samples
Arr_imu=np.array([[0.]*7]*LM)# created stack to remmber LM Last samples
size_of_sample=512
sizeOfGroup=30
channel_from = 2
channel_to = 4
output_path='movement_samples_with_imu2/'
emg_channels=4
imu_channels=6
def imu_current_sample():

    a = imu.get_acceleration()
    b = imu.get_angular_velocity()
    if a and b:
        return np.append(a,b)
    else:
        return np.array([0,0,0,0,0,0])


def collect_samples(size_of_sample, timestamp):
    wait(1)
    array_emg=np.copy(Arr_emg)
    array_imu=np.copy(Arr_imu)
    np.set_printoptions(threshold=sys.maxsize)
    index=np.nonzero((Arr_emg[:, 0] >= timestamp))
    index_of_timestamp=index[0][0]
    emg_re=array_emg[index_of_timestamp-int(size_of_sample/2):index_of_timestamp+int(size_of_sample/2)]
    imu_re=array_imu[index_of_timestamp-int(size_of_sample/2):index_of_timestamp+int(size_of_sample/2)]

    return emg_re,imu_re
    pass


class OPENBCI(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)


    def run(self):
        # resolve an EMG stream on the lab network and notify the user
        print("Looking for an EMG stream...")
        streams = resolve_stream('type', 'EMG')
        inlet = StreamInlet(streams[0])
        print("EMG stream found!")

        while True:
            sample, timestamp = inlet.pull_sample()  # get EMG data sample and its timestamp
            Arr_emg[:-1] = Arr_emg[1:]
            Arr_emg[-1] = np.append([timestamp*10**6],sample)

            Arr_imu[:-1] = Arr_imu[1:]
            Arr_imu[-1] = np.append([timestamp*10**6],imu_current_sample())


        pass
def wait(time_to_wait):# wait a givven time
    time.sleep(time_to_wait)
    pass

def groupData(data,
              size,func):  # loop on data, and group the samples as one sample in size of the size we got in the arguments
    # example: [2,3][4,2][1,2][3,5]=>size=2=>[2,3,4,2][1,2,3,5]
    returnData = np.array([[]])
    for i in range(int(len(data) - size+1)):
        block = func(data,size,i)  # group i to i +size all the data and create new colums for each
        if i == 0:
            returnData = np.append(returnData, block)
        else:
            returnData = np.vstack((returnData, block))

    return returnData

def dictToArray(data):
    re=np.array([])
    for i in data:
        if i != "confidence" and i != "id" and i != "type":
            re=np.append(re,np.array(data[i]).flatten())
    return re



def variance(data,size,i):
    # i = index location , size= how many after i calculate the variance, data= the data to apply on
    return np.var(data[i:i+size], axis=0)



def movementDetected(model_Movement, data, sizeOfGroup):
    clean=np.array([])
    for i in range(len(data)):# remove names and unneeded paramerts from the leap motion data
        sample=dictToArray(data[i]["hands"][0])
        if i==0:
            clean=sample
        else:
            clean=np.vstack((clean,sample))

    data=groupData(clean,sizeOfGroup,variance)
    if model_Movement.predict(np.array([data]))==0:
        return True
    return False
    pass
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

def saveSample(emg_sample, imu_sample):
    global counter
    sample=np.vstack((emg_sample,imu_sample))
    np.savetxt(output_path + str(counter) + '.csv', sample, delimiter=",")
    pass




def to2DArr(data,channel_from,channel_to,resizeTo,reshape_to=4):
    A=np.reshape(data, (-1, reshape_to)).T
    print(len(A[0]),resizeTo)
    A=np.pad(A, ((0, 0), (0, resizeTo-len(A[0]))))

    return A[channel_from:channel_to]
    pass
def remove_timestamp(sample):# removeing timestamps from  the sample

    return np.delete(sample, 0, axis=1)
    pass


counter = 0
async def main(uri):
    isExist = os.path.exists(output_path)
    print(isExist)
    if not isExist:
        os.makedirs(output_path)
    global counter
    counter = LastNumberInFolder(output_path)
    print(counter)
    async with connect(uri) as websocket:

        print("to start press enter")
        x = input()
        print("starting")
        thread2 = OPENBCI()
        thread2.start()
        print("about to start 2 sec")
        model_movement_detect= pickle.load(open(filename, 'rb'))

        wait(1)
        samples_done=1
        livedata_movement=[]
        sizeOfData = 0
        for i in range(100):
            msg = await websocket.recv()
        while True:




            while sizeOfData < sizeOfGroup:  # collecting samples until == size of group
                msg = await websocket.recv()
                sample = json.loads(msg)
                if not sample[
                    "hands"]:  # if sample dosent exists mean the hand wasnt detected and will restart the collection
                    livedata_movement = []
                    sizeOfData = 0
                else:
                    livedata_movement.append(sample)
                    sizeOfData += 1
            print("predict if movement")
            if movementDetected(model_movement_detect,livedata_movement,sizeOfGroup):
                print("movement detected")
                timestamp_leap=livedata_movement[0]['timestamp']
                emg_sample,imu_sample=collect_samples(size_of_sample,timestamp_leap%10**12)
                emg_sample=to2DArr(remove_timestamp(emg_sample),channel_from,channel_to,size_of_sample,reshape_to=emg_channels)
                imu_sample = to2DArr(remove_timestamp(imu_sample), 0, len(imu_sample[0]),size_of_sample,reshape_to=imu_channels)
                saveSample(emg_sample,imu_sample)
                print("sample saved, total:",samples_done)
                samples_done+=1
                for i in range(100):
                    msg = await websocket.recv()
                frequency = 2500  # Set Frequency To 2500 Hertz
                duration = 100  # Set Duration To 1000 ms == 1 second
                winsound.Beep(frequency, duration)
                counter +=1

            livedata_movement = []
            sizeOfData = 0
asyncio.run(main("ws://localhost:6437/v7.json"))
