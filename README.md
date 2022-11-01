![image](https://user-images.githubusercontent.com/51950807/199335380-8748a95b-2942-4402-9145-1d05394388e7.png)



files: 
createDataForLeap.py- simpley record data from the camera,clean unwanted data ,and save it as files, with the wanted class 

createDataForOpenBci.py - create movement samples made of emg signals, make sure either trained models in folder or camera device samples to learn for creating the model

to2Darray.py-reshape 1d array to 2d array in wanted size + fixed to a  consistant size, good for later proccesing

JSONViewer.html- simple html file to see leap motion entire data , fingers cords , etc..

data_proccesing.py - hold all the proccesing the system as to offer, some files use it while procceing data 

live_version.py- live version (unity as to be on), deep learning models as to be in folder,  predict movements in live and send it to unity trow websocket to show in 3d

PLOT.py- plot data channels and show them after and before procceing, good only for testing and preview the data proccesing results 

createDataForLeap.py- simply record leap motion data , put it in folders by class

randomSplit.py- split to train and test a folder full of csv data with classes , for future deep learning and model testing

folders:
extracting python code- contain old files of an older project useing the EMG

learning_EMG -  contain the learning model (deep learning) on the emg data , version 5 is the last, the model that was used for the final result was personal_Net and useing a file called indRNN as a part of is learning model
RNN version 5 architecture:

![image](https://user-images.githubusercontent.com/51950807/179416977-77d4f362-0cfa-4417-b8ab-b597147c3f5a.png)
