

#-----------------------------------------------imports------------------------------------------------#
import asyncio
from websockets import connect
import json
import threading
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from pylsl import StreamInlet, resolve_stream




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
    data = np.loadtxt(open("G:\\git\\final_pro_sem2\\newData\\" + name, "rb"), delimiter=",")
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
    return np.var(data[i:i+size], axis=0)



def studyMovement(dataList_classification, sizeOfGroup):#movement [0] no movement[1]
    X_train, X_valid, y_train, y_valid = loadDataList(dataList_classification, sizeOfGroup,variance)
    depth = 2
    estimators = 50
    alpha =0.003162277660168377

    min_samples_leaf =2
    min_samples_split =2

    clf = RandomForestClassifier(max_depth=depth, n_estimators=estimators, ccp_alpha=alpha,
                                 min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                 random_state=20)
    clf.fit(X_train, y_train)

    print("\n\nmovement detection succeeding rate with RandomForestClassifier:",
          str(compare(clf.predict(X_valid), y_valid) * 100) + "%")


    return clf

    pass


def studyHandGesture(dataList_movement):
    X_train, X_valid, y_train, y_valid = loadDataList(dataList_movement, 1,BlockData)
    depth = 3
    estimators = 72
    alpha =0.003162277660168377

    min_samples_leaf =2
    min_samples_split =2

    clf = RandomForestClassifier(max_depth=depth, n_estimators=estimators, ccp_alpha=alpha,
                                 min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                 random_state=20)
    clf.fit(X_train, y_train)

    print("\n\nclassification succeeding rate with RandomForestClassifier:",
          str(compare(clf.predict(X_valid), y_valid) * 100) + "%")


    return clf
    pass

def movementDetected(model_Movement, data, sizeOfGroup):
    # print("\n\n\nmovementDetected\n\n\n")
    # print(data[0]["currentFrameRate"])
    # print(data[0])
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


def collectFromOpenBciBettwen(from_timestamp, to_timestamp):
    data = np.loadtxt(open("newData/raw/data" + ".txt", 'rb'), delimiter=",")
    # to get a bit of extra data  , by adding to the edges
    # correct_from_timestamp=
    # correct_to_timestamp=
    return data[(data[:,0]>=from_timestamp) & (data[:,0]<=to_timestamp)]
    pass


async def createSampleBettwen(from_timestamp, to_timestamp, movementtype):
    file = open("openbci/" + movementtype +"/data"+ ".txt", 'a+')
    data=collectFromOpenBciBettwen(from_timestamp,to_timestamp)

    file.write(data+'\n')


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

        file = open("openbci/raw/data" + ".txt", 'a+')
        while True:
            sample, timestamp = inlet.pull_sample()  # get EMG data sample and its timestamp
            file.write(timestamp+','+sample)

        pass


#---------------------------------------------------------------------------------------#







async def main(uri):


    async with connect(uri) as websocket:
        # ------------parameters ---------#
        sizeOfData = 0
        sizeOfGroup = 6
        #---------------------------------#

        #---------dataload----------------#
        dataList_movement = [[# data list , data[0]= when y equal to zero , data[1]=when y equal to one
               # "move1.txt",
                "new_movement1.txt",
                "new_movement2.txt",
         ]
         ,[
             # "nomove1.txt",
             # "nomove2.txt",
             # "nomove3.txt",
                "new_nomovement1.txt",
                "new_nomovement2.txt",
                "new_nomovement3.txt",
                "new_nomovement4.txt"
         ]]


        dataList_classification =[[# data list , data[0]= when y equal to zero , data[1]=when y equal to one
            # "rock2.txt",
            # "rock3.txt"
            "new_rock.txt"
         ]
         ,[
            # "paper2.txt",
            # "paper3.txt"
            "new_paper.txt",
            "new_paper2.txt"

         ]
          ,[
            # "scissors2.txt",
            # "scissors3.txt"
            "new_scissors.txt",
            "new_scissors2.txt"
         ]]

        dataTest_classification = ["TESTOPEN.txt", "TESTCLOSE.txt"]  # the data will be only for final testing

        #---------------------------------#

        dataList_classification
        dataList_movement
        model_Movement=studyMovement(dataList_movement,sizeOfGroup)
        model_Classification=studyHandGesture(dataList_classification)


        livedata_movement = []
        thread_data = [1] + [0]
        thread = SummingThread(thread_data)
        thread.start()
        thread2 = OPENBCI(thread_data)
        thread2.start()
        while True:
            toPredict=[]
            movementWasDetected=False

            if thread_data[0]==1:#user on stop
                print("hold")
                while(thread_data[0]!=0):#wait for user respond
                    msg = await websocket.recv()
                print("start")



            #data creation
            while sizeOfData<sizeOfGroup:#collecting samples until == size of group
                # print("about to collect")
                msg=await websocket.recv()
                # print("collected")
                # print(msg)
                sample=json.loads(msg)
                # print(sample["hands"])
                if not sample["hands"]:# if sample dosent exists mean the hand wasnt detected and will restart the collection
                    livedata_movement=[]
                    sizeOfData=0
                else:
                    livedata_movement.append(sample)
                    sizeOfData+=1


            # print("the data:")
            # print(sample)


            if movementDetected(model_Movement,livedata_movement,sizeOfGroup):
                print("movement DETECTED!!!")
                print("create time stamp")
                print(livedata_movement[0]['timestamp'])
                from_timestamp=livedata_movement[0]['timestamp']
                while(movementDetected(model_Movement,livedata_movement,sizeOfGroup)):
                    livedata_movement = []
                    sizeOfData = 0
                    print("still moveing")
                    # data creation
                    while sizeOfData < sizeOfGroup:  # collecting samples until == size of group
                        # print("about to collect")
                        msg = await websocket.recv()
                        # print("collected")
                        # print(msg)
                        sample = json.loads(msg)
                        # print(sample["hands"])
                        if not sample["hands"]:  # if sample dosent exists mean the hand wasnt detected and will restart the collection
                            livedata_movement = []
                            sizeOfData = 0
                        else:
                            livedata_movement.append(sample)
                            sizeOfData += 1
                print("movement stopped!!!")
                print("create time stamp")
                print(livedata_movement[-1]['timestamp'])
                # timestamp=getTimestamp
                to_timestamp = livedata_movement[-1]['timestamp']
                movementtype=model_Classification.predict(np.array([dictToArray(livedata_movement[-1]["hands"][0])]))
                if(movementtype==0):
                    type="rock"
                if(movementtype==1):
                    type="pipper"
                if(movementtype==2):
                    type="scissors"
                print("movement is:",type)
                createSampleBettwen(from_timestamp,to_timestamp,movementtype)
                #save sample
            livedata_movement = []
            sizeOfData = 0


































            #     if sizeOfData<sizeOfGroup:
            #         livedata_movement = []
            #         sizeOfData = 0
            #         break
            #     movementWasDetected=True
            #     msg=await websocket.recv()
            #     sample=json.loads(msg)
            #     print(sample)
            #     if not sample["hands"]:
            #         livedata_movement=[]
            #         sizeOfData=0
            #         break
            #
            #     toPredict+=sample
            #     livedata_movement.append(sample)
            #     livedata_movement = livedata_movement[1:]
            # if movementWasDetected:
            #     DataForOpenBCI[0].append(toPredict[0]["timestamp"])
            #     DataForOpenBCI[1].append(toPredict[-1]["timestamp"])
            #     DataForOpenBCI[2].append(ClassificationPredication(model_Classification,toPredict,HowmanyToPredict,"hands"))
            #     print(DataForOpenBCI[0])
            #     print(DataForOpenBCI[1])
            #     print(DataForOpenBCI[2])
            #     movementWasDetected=False
            #     toPredict=[]
            #     livedata_movement=[]
            #     sizeOfData = 0
            # msg = await websocket.recv()
            # sample= json.loads(msg)["hands"]
            # print(sample)
            # if not sample:
            #     livedata_movement = []
            #     sizeOfData = 0
            #     break
            # else:
            #     livedata_movement += sample
            #     livedata_movement = livedata_movement[1:]

        #
        # jump =0
        # name = input()
        # file = open("newData/"+name +".txt", 'a+')
        # a = [1] + [0]
        # thread = SummingThread(a)
        # thread.start()
        # count = 0
        # while True:
        #     if a[0]==1:
        #         print("hold")
        #         while(a[0]!=0):
        #             msg = await websocket.recv()
        #         print("start")
        #     msg = await websocket.recv()
        #     y = json.loads(msg)
        #     if count==jump:
        #         count=0
        #         if y["hands"]:
        #
        #             obj=y["hands"][0]
        #             toSend=""
        #             for i in obj:
        #                 if i!="confidence" and i!="id" and i!="type":
        #                     toSend+=str(obj[i])+","
        #             toSend=toSend.replace("[","").replace("]","").replace(" ","")
        #             file.write(toSend[:-1] + "\n")
        #     else:
        #         count=count+1

asyncio.run(main("ws://localhost:6437/v7.json"))
