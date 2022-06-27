import asyncio
import os

from websockets import connect
import json
import threading
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from pylsl import StreamInlet, resolve_stream

record_name="for_mvc.txt"
path="free_records/"
ignore_channels=[0,1]


# resolve an EMG stream on the lab network and notify the user
print("Looking for an EMG stream...")
streams = resolve_stream('type', 'EMG')
inlet = StreamInlet(streams[0])
print("EMG stream found!")
print("press enter to start")
size=2000
_=input()
if not os.path.exists(path):
    os.makedirs(path)
file = open(path+record_name, 'a+')

counter_save=0
counter=0
while(counter<size):
    sample, timestamp = inlet.pull_sample()  # get EMG data sample and its timestamp
    arr=[]
    for i in range(len(sample)):
        if i not in ignore_channels:
            arr.append(sample[i])
    to_send=str(arr).replace("[","").replace("]","").replace(" ","")
    file.write(to_send+"\n")
    counter+=1
    if counter%100==0:
        print(counter/100)
    pass