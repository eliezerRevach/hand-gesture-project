import asyncio
from websockets import connect
import time

async def hello(uri):
    async with connect(uri) as websocket:

        while True:
            msg=await websocket.recv()

            



asyncio.run(hello("ws://localhost:6437/v7.json"))