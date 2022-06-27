import sys
import threading
import time

import websockets

import data_proccesing as dp
from pylsl import StreamInlet, resolve_stream
import numpy as np
import torch.nn as nn
import torch
from indrnn import IndRNN
from witmotion import IMU

LM=50000#Last to remmber, an array that openbci keep feilling and droping the last values when full
size_of_movement_frame_after_rms=450#movement sample size after proccesing
size_of_movement=512# raw data size of movement sample
window_size=50# window size for rms
extra_samples=500# extra samples to start with with raw data for predict hand gesture
imu = IMU(path="COM8")# usb port of IMU
ignore_channels=[0,1]# emg channels wanted to ingonre/ not in use

std_from_multipli=1.5#mean - std*multipli... -> start of sample
std_to_multipli=2#mean + std*multipli... -> end of sample
var=True# use var for normal distrubation
var_size=50# size of frame for var in normal distrubtuion
time_to_wait=2# time delay before predicting the gesture // higher better prediction
raw_const_size_movement=512# size of the window frame to detect movement
raw_const_size_hand_gesture=512*3# raw size its getting
size_of_gesture_final=512*2# final size getting into the nn for gesture prediction
normalize=False# normalizing the data by the mvc
NUM_OF_CHANNELS=4# number of channel of the openbci device
reshape_to=4# reshape the samples of each channel to 2d array ,sould be equal to number of channels to
wanted_size_of_mvc=5000# how many samples to collect when creating mvc
imu_channels=6
#---- nn params
device = torch.device('cpu')
#RNN movement model
path_to_movement_model="model_rnn"
RNN_layers=3
RNN_Classes=2
sequence_length=1
#IRNN type model
certain_state_mod=True
path_to_gesture_model="model_irnn"
default="rock"
path_to_rock_state_model="model_irnn_rock"
path_to_paper_state_model="model_irnn_paper"
path_to_scissors_state_model="model_irnn_scissors"

IRNN_layers=3
IRNN_Classes=3
time_steps=100
batch_norm=True
bidirectional=True





RECURRENT_MAX = pow(2, 1 / time_steps)

#models :
#type model IRNN
class Net(nn.Module):# IRNN model , using IRNN then useing FC
    def __init__(self, input_size, hidden_size, n_layer=2, model=IndRNN):
        super(Net, self).__init__()
        recurrent_inits = [lambda w: nn.init.uniform_(
            w, -RECURRENT_MAX, RECURRENT_MAX)]
        for _ in range(1, n_layer):
            recurrent_inits.append(lambda w: nn.init.constant_(w, 1))
        self.indrnn = model(
            input_size, hidden_size,
            n_layer, batch_norm=batch_norm,
            bidirectional=bidirectional,
            hidden_max_abs=RECURRENT_MAX,
            recurrent_inits=recurrent_inits)
        self.lin = nn.Linear(
            hidden_size * 2 if bidirectional else hidden_size, 1)
        self.fc = nn.Linear( hidden_size * 2 if bidirectional else hidden_size, IRNN_Classes)
        self.fc.bias.data.fill_(.1)
        self.fc.weight.data.normal_(0,.01)
    def forward(self, x, hidden=None):
        y, _ = self.indrnn(x, hidden)
        return self.fc(y[:, -1, :])
        # return self.lin(y[-1]).squeeze(1)

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out




#---------------------------------------------------------------------------------------------------------------------#
normalize_array=None
Arr=np.array([[0.]*5]*LM)# created stack to remmber LM Last samples
Arr_imu=np.array([[0.]*7]*LM)# created stack to remmber LM Last samples

def imu_current_sample():

    a = imu.get_acceleration()
    b = imu.get_angular_velocity()
    if a and b:
        return np.append(a,b)
    else:
        return np.array([0,0,0,0,0,0])


class OPENBCI(threading.Thread):
    def __init__(self,mon):
        threading.Thread.__init__(self)
        self.arr=mon

    def run(self):
        # resolve an EMG stream on the lab network and notify the user
        print("Looking for an EMG stream...")
        streams = resolve_stream('type', 'EMG')
        inlet = StreamInlet(streams[0])
        print("EMG stream found!")

        while True:
            sample, timestamp = inlet.pull_sample()  # get EMG data sample and its timestamp
            Arr[:-1] = Arr[1:]
            Arr[-1] = np.append([timestamp],sample)
            Arr_imu[:-1] = Arr_imu[1:]
            Arr_imu[-1] = imu_current_sample()


        pass


def collect_last_data(size_of_movement):# copyinmg the array and getting is last data frame in the given size
    array=np.copy(Arr)
    return (array[len(array)-size_of_movement:])
    pass
def collect_last_data_from_imu(size_of_movement):# copyinmg the array and getting is last data frame in the given size
    array=np.copy(Arr_imu)
    return (array[len(array)-size_of_movement:])
    pass
def get_first_timestamp(sample):# getting the timestamp of the first one in the array
    return sample[0][0]
    pass

def TwoDof(sample, reshape_to, ignore_channels,const_size=-1):
    """
    getting a array, made of few channels ,
    reshaping to the number of channels as the number of rows,
    and removeing unwanted ones,then adding zeros to complete to a const size
    for all the samples , for future proccesing
    """
    A=sample[:, :].flatten()
    A=np.reshape(A, (-1, reshape_to)).T
    if const_size !=-1:
        A = np.pad(A, ((0, 0), (0, const_size - len(A[0]))))
    if ignore_channels:
        A=np.delete(A,ignore_channels,axis=0)
    return A
    pass
def remove_timestamp(sample):# removeing timestamps from  the sample

    return np.delete(sample, 0, axis=1)
    pass

def get_last_timestamp(sample):# getting the timestamp int the end of the array
    return sample[len(sample)-1][0]
    pass


def wait(time_to_wait):# wait a givven time
    time.sleep(time_to_wait)
    pass


def collectDataBettwen(first_timestamp, last_timestamp, extra_samples):# get's all the samples bettewn the 2 timestamp incudeing them
    array=np.copy(Arr)
    print(first_timestamp, last_timestamp)
    np.set_printoptions(threshold=sys.maxsize)

    index = np.nonzero((array[:,0]>=first_timestamp) & (array[:,0]<=last_timestamp))

    index_a = max(index[0][0] - extra_samples,0)
    index_b = min(index[0][len(index[0]) - 1] + extra_samples,len(array)-1)
    return array[index_a:index_b+1]
    pass


def process_movement(sample, final_size, window_size=False,normalize=False):# do rms on the movement sample
    if normalize:
        return dp.toRMS(sample,normalize_array,window_size=window_size,complete_to_final=final_size)
    return dp.toRMS(sample,window_size=window_size,complete_to_final=final_size)
    pass


def process_gesture(sample, raw_size,final_size, window_size,normalize):# do normal distribution to get the borders using mean and std of the sample  and then rms
    mean_arr,std_arr=dp.twoD_normal_distribution(sample,var_farme_size=var_size,var=var)
    cutted_by_normal_distribution=dp.twoD_cut_by_std(sample,mean_arr,std_arr,std_from_multipli,std_to_multipli,raw_size)
    if normalize:
        return dp.toRMS(cutted_by_normal_distribution,normalize_array,window_size=window_size,complete_to_final=final_size)
    return dp.toRMS(cutted_by_normal_distribution,window_size=window_size,complete_to_final=final_size)
    pass


def get_input_size_of_RNN():#calculate the input size for the rnn model depending on the number of channels
    return size_of_movement_frame_after_rms*(NUM_OF_CHANNELS-len(ignore_channels))
    pass


def get_hidden_size_of_RNN():#calculate the hidden size for the rnn model depending on the number of channels
    return get_input_size_of_RNN()
    pass


def np_to_torch(sample):# take a numpy array , and transforming it to torch array for nn
    my_x = []
    my_x.append(sample)
    return torch.Tensor(my_x)
    pass


def predict_movement(sample,model,imu_sample):
    combined_sample=np.vstack((sample,imu_sample))
    model.eval()
    outputs = model(np_to_torch(combined_sample).reshape(-1, sequence_length, get_input_size_of_RNN()).to(device))
    _, predicted = torch.max(outputs.data, 1)
    print(predicted[0])
    if predicted[0]==1:
        return True
    return False
    pass


def get_input_size_of_IRNN():
    return size_of_gesture_final*(NUM_OF_CHANNELS-len(ignore_channels))
    pass
def get_hidden_size_of_IRNN():
    return get_input_size_of_IRNN()
    pass


def predict_gesture(sample,model):
    model.eval()
    outputs = model(np_to_torch(sample).reshape(-1, sequence_length, get_input_size_of_IRNN()).to(device))
    _, predicted = torch.max(outputs.data, 1)
    print(predicted)
    return predicted[0]
    pass


def To_MVC(data):# get a 2d array ,and callcoule the mean of each array , used for mvc
    return np.mean(np.abs(data),axis=0)


def collectXSamplesfromBCI(wanted_size_of_mvc): # call openbci to start collecting dat until it reach the wanted size
    # resolve an EMG stream on the lab network and notify the user
    print("Looking for an EMG stream...")
    streams = resolve_stream('type', 'EMG')
    inlet = StreamInlet(streams[0])
    print("EMG stream found!")
    returned_array=[]
    for i in range(wanted_size_of_mvc):
        sample, timestamp = inlet.pull_sample()  # get EMG data sample and its timestamp
        returned_array.append(np.array(sample))
        if i%(wanted_size_of_mvc/10)==0:
            print(10-i/(wanted_size_of_mvc/10))
    return returned_array
    pass


def load_model(path_to_gesture_model, param):# loading a model based on path ,and the type of the model
    if param=="RNN":
        model = RNN(get_input_size_of_RNN(), get_hidden_size_of_RNN(), RNN_layers, RNN_Classes).to(device)
    if param=="IRNN":
        model = Net(get_input_size_of_IRNN(), get_hidden_size_of_IRNN(), IRNN_layers).to(device)
    model.load_state_dict(torch.load(path_to_gesture_model,map_location=torch.device('cpu')))
    model.eval()
    return model
    pass


def defult_model(model_gesture_rock, model_gesture_paper, model_gesture_scissors):
    if default=="rock":
        return model_gesture_rock
    if default=="paper":
        return model_gesture_paper
    if default=="scissors":
        return model_gesture_scissors
    pass


def main():
    if certain_state_mod:
        model_gesture_rock=load_model(path_to_rock_state_model,"IRNN")
        model_gesture_paper = load_model(path_to_paper_state_model, "IRNN")
        model_gesture_scissors = load_model(path_to_scissors_state_model, "IRNN")
        model_gesture=defult_model(model_gesture_rock,model_gesture_paper,model_gesture_scissors)
        current_model=default
    else:
        model_gesture=load_model(path_to_gesture_model,"IRNN")

    model_movement = load_model(path_to_movement_model,"RNN")
    if normalize:
        print("press enter to MVC start, use max force until it will tell u to stop, it will take around 10 sec ")
        x=input()
        data=collectXSamplesfromBCI(wanted_size_of_mvc)
        global normalize_array
        normalize_array= To_MVC(data)
    print("to start press enter")
    x = input()
    thread2 = OPENBCI(Arr)
    thread2.start()
    wait(2)

    while True:
        r"""
        running with a window frame in a given size watching the last samples from the openbci
        and predicting if there was a movement,
        we collect the data in of the frame , reshping to a 2d array removeing the timestamp ,
        proccsing as a movement data process and predicing if there was a movement
        """
        sample=collect_last_data(size_of_movement)
        imu_sample =TwoDof(collect_last_data_from_imu(size_of_movement),imu_channels,None)
        first_timestamp = get_first_timestamp(sample)
        sample=process_movement(TwoDof(remove_timestamp(sample),reshape_to,ignore_channels,const_size=raw_const_size_movement),size_of_movement_frame_after_rms,window_size,normalize)

        if predict_movement(sample,model_movement,imu_sample):
            while predict_movement(sample,model_movement):
                sample = collect_last_data(size_of_movement)
                last_timestamp = get_last_timestamp(sample)
                sample = process_movement(TwoDof(remove_timestamp(sample),reshape_to,ignore_channels,const_size=raw_const_size_movement), size_of_movement_frame_after_rms,window_size,normalize)
            print("waiting 1 sec")
            wait(time_to_wait)

            r"""
            collect the sample bettwen the 2 timestamp with extra x amount from before and after,
            then reshaping to a 2d array and remove the timestamps from the sample
            then proccesing the data we have and predictin the gesture
            """
            sample=collectDataBettwen(first_timestamp,last_timestamp,extra_samples)
            print(sample.shape)
            in2d=TwoDof(remove_timestamp(sample),reshape_to,ignore_channels,const_size=raw_const_size_hand_gesture)
            print(in2d.shape)
            sample=process_gesture(in2d,raw_const_size_hand_gesture,size_of_gesture_final,window_size,normalize)
            results=predict_gesture(sample,model_gesture)
            if certain_state_mod:
                if current_model == "rock":
                    if results == 0:
                        send_gesture("paper")
                        current_model="paper"
                        model_gesture=model_gesture_paper

                    elif results == 1:
                        send_gesture("scisssors")
                        current_model="scisssors"
                        model_gesture=model_gesture_scissors

                elif current_model == "paper":
                    if results == 0:
                        send_gesture("rock")
                        current_model="rock"
                        model_gesture=model_gesture_rock

                    elif results == 1:
                        send_gesture("scisssors")
                        current_model="scisssors"
                        model_gesture=model_gesture_scissors

                elif current_model == "scisssors":
                    if results == 0:
                        send_gesture("rock")
                        current_model="rock"
                        model_gesture=model_gesture_rock

                    elif results == 1:
                        send_gesture("paper")
                        current_model="paper"
                        model_gesture=model_gesture_paper
            else:
                if results==0:
                    send_gesture("rock")
                elif results==1:
                    send_gesture("paper")
                else:
                    send_gesture("scissors")
            wait(1)

def send_gesture(name):# send to unity hand gesture
    async with websockets.connect("ws://127.0.0.1:7891/gesture") as websocket:
        await websocket.send(name)

main()