

#-----------------------------------------------imports------------------------------------------------#
import asyncio
import os

from websockets import connect
import json
import threading
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from pylsl import StreamInlet, resolve_stream

Folder_leap_data="G:\\git\\final_pro_sem2\\old_dir\\newData\\"
lock = threading.Lock()
realse_lock_after = 30# will save the data into the text after x samples given

#------------------------------------------------------------------------------------------------------#

#-------------------------------------------functions--------------------------------------------------#


#-------------------------------actions on data--------------------------------------------#
def loadDataList(dataList,
                 sizeOfGroup,func):#  # get a list of data to learn(data[0]=when y = 0,data[1]=when y = 1),with the size of the group wanted,return X_train, X_valid, y_train, y_valid
                    #dataList=files, size of group = how many samples to group into one, like grouping with variance,func = like varaince
    data = np.array(None)
    labels = np.array(None)
    for i in range(len(dataList)):
        for j in range(len(dataList[i])):
            data_temp, labels_temp = load_single_data(dataList[i][j], sizeOfGroup, 20, i,func)
            if data.any() == None:
                data = data_temp
                labels = labels_temp
            else:
                data = np.vstack((data, data_temp))
                labels = np.append(labels, labels_temp)

    X_train, X_valid, y_train, y_valid = train_test_split(data, labels, test_size=0.1, random_state=20, shuffle=True)

    return X_train, X_valid, y_train, y_valid


def BlockData(data,size,i):
    # i = index location , size = how many to block, data = the data to apply on
    return np.array(data[i:i + size].flatten())


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

def load_single_data(name,sizeOfGroup,split,output,func):  # get data name with y =0,data name with y=1, how many lines to group for each sample and the size to cut the edges
    #name = name of file
    #sizeOfGroup= how many to group
    #split for removeing edges, remove the size of data*2(each edge)/split
    #i=0 or 1  for the labels , if itmovement or not
    #func= func type like variance
    data = np.loadtxt(open(Folder_leap_data + name, "rb"), delimiter=",")
    data = data[int(len(data) / split):int(len(data) - len(data) / split)]
    data = groupData(data, sizeOfGroup,func)
    labels = np.array([output] * len(data))
    return data, labels


def cleanData(data, param):
    return data[:,param].replace("[","").replace("]","").replace(" ","")
    pass
#---------------------------------------------------------------------------------------#













#-------------------------------predictions---------------------------------------------#
def predictWithSum(predicted,
                   k):  # check the last k results in the predict in each step and go by what most say , if 1 or 0 and return it ,
    # example with k=5 : 0000 0001 0011 0101 predict, will return 0000 0000 0001 1111 ,make it more sequential
    re = np.array([])
    for i in range(k):
        re = np.append(re, predicted[i])

    sumOf1 = np.sum(re[0:k] == 1)
    sum = 2 * sumOf1 - k  # sumOf1-(len(sumOf1)-sumOf1) how many 1 minues how many not 1
    for i in range(len(predicted) - k):
        j = i + k
        sum, decision = sumLastKSize(predicted[j], predicted[j - k], sum)
        re = np.append(re, decision)

    return re


def sumLastKSize(add, remove,
                 sum):  # calculte the next sum size k of the next sample(by removeing the first and adding a new one in order)
    # example :[00110]0=> sum=-1, sum-(-1)+-1->sum=-1 => 0[01100]
    decision = 0
    if remove == 0:
        remove = -1
    if add == 0:
        add = -1
    sum = sum + add - remove
    if sum < 0:
        decision = 0
    else:
        decision = 1
    return sum, decision
def variance(data,size,i):
    # i = index location , size= how many after i calculate the variance, data= the data to apply on
    return np.var(data[i:i+size], axis=0)



def studyMovement(dataList_classification, sizeOfGroup):#movement [0] no movement[1]
    filename = 'model_movement.sav'
    X_train, X_valid, y_train, y_valid = loadDataList(dataList_classification, sizeOfGroup, variance)
    if not file_exists(filename):
        print("creating movement model")


        depth = 19
        estimators = 260
        alpha =0.000001

        min_samples_leaf =12
        min_samples_split =12

        clf = RandomForestClassifier(max_depth=depth, n_estimators=estimators, ccp_alpha=alpha,
                                     min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                     random_state=20)
        clf.fit(X_train, y_train)
        pickle.dump(clf, open(filename, 'wb'))
    else:
        print("loading movement model\n...")
        clf = pickle.load(open(filename, 'rb'))
    print("nmovement detection succeeding rate with RandomForestClassifier:",
          str(compare(clf.predict(X_valid), y_valid) * 100) + "%")


    return clf

    pass

import pickle
from os.path import exists as file_exists
def studyHandGesture(dataList_movement):
    filename = 'model_HandGesture.sav'
    X_train, X_valid, y_train, y_valid = loadDataList(dataList_movement, 1, BlockData)

    if not file_exists(filename):
        print("creating hand gesture model")
        depth = 11
        estimators = 160
        alpha =0.001

        min_samples_leaf =4
        min_samples_split =4

        clf = RandomForestClassifier(max_depth=depth, n_estimators=estimators, ccp_alpha=alpha,
                                     min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                     random_state=20)
        clf.fit(X_train, y_train)

        pickle.dump(clf, open(filename, 'wb'))
    else:
        print("loading hand gesture model")

        clf=pickle.load(open(filename, 'rb'))
    print("\n...\nclassification succeeding rate with RandomForestClassifier:",
          str(compare(clf.predict(X_valid), y_valid) * 100) + "%")


    return clf
    pass

def movementDetected(model_Movement, data, sizeOfGroup):
    clean=np.array([])
    for i in range(len(data)):
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


def ClassificationPredication(model_Classification, toPredict, HowmanyToPredict, param):#HowmanyToPredict not installed
    clean=cleanData(toPredict,param)
    predicted=model_Classification.predict(clean)
    return max(set(predicted), key=predicted.count)

    pass

#---------------------------------------------------------------------------------------#

def compare(a, b):  # return how many was right divide by all
    size = len(a)
    sum = np.sum(a == b)
    return sum / size


def dictToArray(data):
    re=np.array([])
    for i in data:
        if i != "confidence" and i != "id" and i != "type":
            re=np.append(re,np.array(data[i]).flatten())
    return re


def collectFromOpenBciBettwen(from_timestamp, to_timestamp,offset_left=0,offset_right=0):
    # print("test:",from_timestamp)
    data = np.loadtxt(open("openbci/raw/data" + ".txt", 'rb'), delimiter=",")
    index = np.nonzero((data[:,0]>=from_timestamp) & (data[:,0]<=to_timestamp))
    # print("current raw file size:",len(data))
    # print("index len:",len(index[0]))
    # print("original from",index[0][0])
    # print("original after",index[0][len(index[0]) - 1])
    # print("off left",offset_left)
    # print("off right",offset_right)
    index_a = max(index[0][0] - offset_left,0)
    index_b = min(index[0][len(index[0]) - 1] + offset_right,len(data)-1)
    return data[index_a:index_b+1]
    pass


def createSampleBettwen(from_timestamp, to_timestamp, movementtype,symbol,offset_left=0,offset_right=0,foldering_by_last=False,last=None):
    path = "openbci/" + str(movementtype[0]) + "/"
    if foldering_by_last:
        path = "openbci/with_last/"+str(last)+"/"+ str(movementtype[0])+"/"
    if not os.path.exists(path):
        os.makedirs(path)
    file = open(path + "data" +symbol + ".txt", 'a+')
    from_=from_timestamp
    to_=  to_timestamp
    # print("test",offset_left)

    data=collectFromOpenBciBettwen(from_,to_,offset_left=offset_left,offset_right=offset_right)
    import sys

    np.set_printoptions(threshold=sys.maxsize)
    # print("data len:",len(data))
    #cleaning the data for a string to write in file leaveing only the samples without the timestamps
    to_send=np.array2string(data[:, 1:].flatten(), separator=',').replace("[", "").replace("]", "").replace(" ", "").replace("\n", "")
    file.write(to_send+'\n')


    pass
def cleanData():#remove the data from data up to timestamp to avoid files to heavy
    # print("getting lock to clean the data ")
    lock.acquire()
    file = open("openbci/raw/data" + ".txt", 'a+')
    file.seek(0)
    file.write("")
    file.truncate()
    file.close()
    lock.release()
    pass

def print_results(type, from_timestamp, to_timestamp, last_type, foldering_by_last):
    import pandas
    if foldering_by_last:
        data=[last_type,type,str(from_timestamp),str(to_timestamp)]
        list_names=["last hand gesture","new hand gesture","first timestamp","sec timestamp"]

    else:
        data=[type,str(from_timestamp),str(to_timestamp)]
        list_names=["hand gesture","first timestamp","sec timestamp"]
    pandas.DataFrame(data, list_names, "sample")
    pass

#------------------------------------------------------------------------------------------------------#





#-------------------------------------thread---------------------------------------------#
class SummingThread(threading.Thread):
    def __init__(self, mon):
        threading.Thread.__init__(self)
        self.mon = mon

    def run(self):

        while True:
            user = input()
            self.mon[0]=0
            user = input()
            self.mon[0] =1



class OPENBCI(threading.Thread):
    def __init__(self, mon):
        threading.Thread.__init__(self)
        self.mon = mon

    def run(self):
        # resolve an EMG stream on the lab network and notify the user
        print("Looking for an EMG stream...")
        streams = resolve_stream('type', 'EMG')
        inlet = StreamInlet(streams[0])
        print("EMG stream found!")

        file = open("openbci/raw/data.txt", 'a+')
        counter_save=0
        while True:

            sample, timestamp = inlet.pull_sample()  # get EMG data sample and its timestamp
            to_send=str(sample).replace("[","").replace("]","").replace(" ","")
            if counter_save == 0:
                lock.acquire()

            file.write(str(timestamp*10**6)+','+to_send+"\n")
            counter_save+=1

            if realse_lock_after==counter_save:
                lock.release()
                counter_save = 0
                pass

        pass

#---------------------------------------------------------------------------------------#






async def main(uri):
    # ------------parameters ---------#

    sizeOfGroup =30# for the leap motion movement detection , grouping x amout of data and check if it was a movement
    foldering_by_last=True
    remmber_before=500
    remmber_after=500
    to_leave=300
    # ---------------------------------#
    if foldering_by_last:
        last=0
    last_type="None"
    symbol=0
    sizeOfData = 0
    async with connect(uri) as websocket:

        #---------dataload----------------#
        dataList_movement = [[# data list , data[0]= when y equal to zero , data[1]=when y equal to one
            "move1.txt",
            "move2.txt",
            "move3.txt",
            "move4.txt",

         ]
         ,[
             "nomove1.txt",
             "nomove2.txt",
             "nomove3.txt",
             "nomove4.txt",
             "nomove5.txt",
             "nomove6.txt",
             "nomove7.txt",
             "nomove8.txt",

         ]]


        dataList_classification =[[# data list , data[0]= when y equal to zero , data[1]=when y equal to one # ORDER ROCK PAPER SCISSORS
            "rock1.txt",
            "rock2.txt",
            "rock3.txt",
            "rock4.txt",
            "rock5.txt",

         ]
         ,[
            "papper1.txt",
            "pepper2.txt",
            "pepper3.txt",
            "pepper4.txt",


         ]
          ,[
            "scissors1.txt",
            "scissors2.txt",
            "scissors3.txt",
            "scissors4.txt",

         ]]

        #---------------------------------#

        model_Movement=studyMovement(dataList_movement,sizeOfGroup)
        model_Classification=studyHandGesture(dataList_classification)


        livedata_movement = []
        thread_data = [1] + [0]
        import os
        if os.path.exists("openbci/raw/data.txt"):
            os.remove("openbci/raw/data.txt")
        if not os.path.exists("openbci/raw/"):
            os.makedirs("openbci/raw/")
        thread = SummingThread(thread_data)
        thread.start()
        thread2 = OPENBCI(thread_data)
        thread2.start()
        counter=0
        while True:

            if thread_data[0]==1:#user on stop
                print("press enter to start")
                while(thread_data[0]!=0):#wait for user respond
                    msg = await websocket.recv()
                print("starting...")
                symbol+=1



            #data creation

            while sizeOfData<sizeOfGroup:#collecting samples until == size of group
                msg=await websocket.recv()
                sample=json.loads(msg)
                if not sample["hands"]:# if sample dosent exists mean the hand wasnt detected and will restart the collection
                    livedata_movement=[]
                    sizeOfData=0
                else:
                    livedata_movement.append(sample)
                    sizeOfData+=1

            if movementDetected(model_Movement,livedata_movement,sizeOfGroup):
                print("movement DETECTED!!!")
                # print("create time stamp")
                print(livedata_movement[0]['timestamp'])
                from_timestamp=livedata_movement[0]['timestamp']
                while(movementDetected(model_Movement,livedata_movement,sizeOfGroup)):
                    livedata_movement = []
                    sizeOfData = 0
                    # print("still moving")
                    # data creation
                    while sizeOfData < sizeOfGroup:  # collecting samples until == size of group
                        msg = await websocket.recv()
                        sample = json.loads(msg)
                        if not sample["hands"]:  # if sample dosent exists mean the hand wasnt detected and will restart the collection
                            livedata_movement = []
                            sizeOfData = 0
                        else:
                            livedata_movement.append(sample)
                            sizeOfData += 1
                # print("movement stopped!!!")
                # print("create time stamp")
                # print(livedata_movement[-1]['timestamp'])
                # timestamp=getTimestamp
                to_timestamp = livedata_movement[-1]['timestamp']
                movementtype=model_Classification.predict(np.array([dictToArray(livedata_movement[-1]["hands"][0])]))
                if(movementtype==0):
                    type="rock"
                if(movementtype==1):
                    type="pipper"
                if(movementtype==2):
                    type="scissors"
                # print("movement is:",type)# here add list
                print_results(type,from_timestamp,to_timestamp,last_type,foldering_by_last)
                import time
                # wait a bit after movement end to get extra data
                print("Please wait 5 sec before moving hand.")
                for i in range(3):
                    print(5-i)
                    time.sleep(1)
                if foldering_by_last:
                    # print("last movement was:",last_type)
                    createSampleBettwen(from_timestamp % 10 ** 12, to_timestamp % 10 ** 12, movementtype, str(symbol),offset_left=remmber_before,offset_right=remmber_after,foldering_by_last=foldering_by_last,last=last)
                    last=movementtype[0]
                    last_type=type
                else:
                    createSampleBettwen(from_timestamp%10**12,to_timestamp%10**12,movementtype,str(symbol),offset_left=remmber_before,offset_right=remmber_after)
                # print("cleaning")
                cleanData()
                # print("cleaned")
                counter+=1
                # print("samples created this run:", counter)
                if counter==to_leave:
                    exit()
                # print("wait 2 sec")
                import time
                for i in range(2):
                    print(2-i)
                    time.sleep(1)
                import winsound
                frequency = 2500  # Set Frequency To 2500 Hertz
                duration = 100  # Set Duration To 1000 ms == 1 second
                winsound.Beep(frequency, duration)
                print("Ridy")
                #save sample
            livedata_movement = []
            sizeOfData = 0

asyncio.run(main("ws://localhost:6437/v7.json"))
