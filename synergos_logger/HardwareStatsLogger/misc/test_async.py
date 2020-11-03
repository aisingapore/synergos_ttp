# import asyncio

# async def myCoroutine():
#     print("My Coroutine")

# loop = asyncio.get_event_loop()
# try:
#     loop.run_until_complete(myCoroutine())
# finally:
#     loop.close()

import asyncio
import functools
import os
import signal
import psutil
import time

# def ask_exit(signame, loop):
#     print("got signal %s: exit" % signame)
#     loop.stop()

# async def some_func(): # logging
#     for i in range(1000000):
#         print(i)

# async def main():
#     loop = asyncio.get_running_loop()

#     for signame in {'SIGINT', 'SIGTERM'}:
#         loop.add_signal_handler(
#             getattr(signal, signame),
#             functools.partial(ask_exit, signame, loop))
#     # for i in range(1000000):
#     #     print('z ' + str(i))

#     await asyncio.sleep(5)
#     # await some_func()

# print("Event loop running for 1 hour, press Ctrl+C to interrupt.")
# print(f"pid {os.getpid()}: send SIGINT or SIGTERM to exit.")

# asyncio.run(main())
# some_func()




# async def say_after(delay, what):
#     # await asyncio.sleep(delay)
#     for i in range(500000):
#         print(what + str(i))
#     # print(what)

# async def main():
#     task1 = asyncio.create_task(
#         say_after(1, 'hello'))

#     task2 = asyncio.create_task(
#         say_after(2, 'world'))

#     print(f"started at {time.strftime('%X')}")

#     # Wait until both tasks are completed (should take
#     # around 2 seconds.)
#     await task1
#     await task2

#     print(f"finished at {time.strftime('%X')}")

# asyncio.run(main())




import asyncio

async def first():
    await asyncio.sleep(5)
    return "1"

async def second():
    await asyncio.sleep(5)
    return "2"

# async def hwlogger():
#     while True:
#         print(psutil.cpu_percent(interval=1))

async def main():
    async def one_iteration():
        result = await first()
        print(result)
        result2 = await second()
        print(result2)
    coros = [one_iteration() for _ in range(1)]
    await asyncio.gather(*coros)
    # for i in range(2):
    #   result = await first()
    #   print(result)
    #   result2 = await second()
    #   print(result2)


loop = asyncio.get_event_loop()
# # while True:
# #     print(psutil.cpu_percent(interval=5))
loop.run_until_complete(main())

# import HardwareStatsLogger
# HardwareStatsLogger.main()