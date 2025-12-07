import asyncio
import websockets
import json

async def main():
    url="ws://127.0.0.1:5001/ws?token=MyS3CR37C0D3"
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"cmd":"dmesg | grep error"}))

        async for message in ws:
            data = json.loads(message)
            print(data)
        

asyncio.run(main())