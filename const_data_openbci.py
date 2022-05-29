#-----------------------------------------------imports------------------------------------------------#
import os

import numpy as np
from pylsl import StreamInlet, resolve_stream

#------------------------------------------------------------------------------------------------------#


#--------1d array to 2d array-----------#



def to2DArr(data,channel_from,channel_to,resizeTo):
    A=np.reshape(data, (-1, 4)).T
    A=np.pad(A, ((0, 0), (0, resizeTo-len(A[0]))))

    return A[channel_from:channel_to]
    pass

#-------------------------------------#

def LastNumberInFolder(path):
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


def openbci_stream(sample_size,output, channel_from, channel_to, resizeTo):
    print("Looking for an EMG stream...")
    streams = resolve_stream('type', 'EMG')
    inlet = StreamInlet(streams[0])
    print("EMG stream found!")

    counter=LastNumberInFolder(output)
    while True:
        counter += 1
        data=""
        for i in range(sample_size):
            sample, timestamp = inlet.pull_sample()  # get EMG data sample and its timestamp
            data+= str(sample).replace("[", "").replace("]", "").replace(" ", "")+","

        to_send=to2DArr(data, channel_from, channel_to, resizeTo)
        np.savetxt(output + str(counter) + '.csv', to_send, delimiter=",")


def main():

    #params:
    sample_size=200
    output_path="rock_paper_scissor_430size_RMS_CSV/0/"
    channel_from=2
    channel_to=4
    resizeTo=200
    #-----------------------#

    isExist = os.path.exists(output_path)
    if not isExist:
        os.makedirs(output_path)
    openbci_stream(sample_size,output_path, channel_from, channel_to, resizeTo)

main()
