import asyncio
import threading
from lib2to3.pgen2 import driver


from websockets import connect
import time
import json

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


async def hello(uri):


    async with connect(uri) as websocket:
        jump =0
        print("name")
        name = input()
        file = open("newData/"+name +".txt", 'a+')
        a = [1] + [0]
        thread = SummingThread(a)
        thread.start()
        count = 0
        while True:
            if a[0]==1:
                print("hold")
                while(a[0]!=0):
                    msg = await websocket.recv()
                print("start")
            msg = await websocket.recv()
            y = json.loads(msg)
            if count==jump:
                count=0
                if y["hands"]:

                    obj=y["hands"][0]
                    toSend=""
                    for i in obj:
                        if i!="confidence" and i!="id" and i!="type":
                            toSend+=str(obj[i])+","
                    toSend=toSend.replace("[","").replace("]","").replace(" ","")
                    file.write(toSend[:-1] + "\n")
            else:
                count=count+1

asyncio.run(hello("ws://localhost:6437/v7.json"))