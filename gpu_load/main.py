# -*- coding: utf-8 -*-
#!/usr/bin/python                        
##################################################
# AUTHOR : Yandi LI
# CREATED_AT : 2018-11-01
# LAST_MODIFIED : 2018-11-07 12:55:32
# USAGE : python -u main.py
# PURPOSE : GPU占用程序
##################################################
from __future__ import division
import math
import threading
import time
from collections import deque
import GPUtil
from numba import cuda
import numpy

cuda.select_device(0)

class Monitor(threading.Thread):
  """ 后台检测当前GPU占用率
  """

  def __init__(self):
    super(Monitor, self).__init__()
    self.setDaemon(True)
    self._queue = deque([0] * 5, 5)
    self.avg_load = 0
    self.max_load = 0

  def update(self, ):
    load = self.get_current_load()
    self._queue.append(load)
    self.avg_load = sum(self._queue)/len(self._queue)
    self.max_load = max(self._queue)

  def run(self):
    while True:
      self.update()
      time.sleep(1)

  @staticmethod
  def get_current_load():
    gpu = GPUtil.getGPUs()[0]
    load = gpu.load * 100
    return load


class Worker(object):
  """ GPU占用程序
  - 根据目标target，自动调整需要用到的blocks数量
  - 如果monitor检测有其他程序争抢GPU，峰值超过阈值，则自动切断运行
  """

  def __init__(self, target=50):
    data = numpy.zeros(512)
    self._device_data = cuda.to_device(data)
    self.threadsperblock = 128
    self.blockspergrid = int(math.ceil(data.shape[0] / self.threadsperblock)) 
    self.target = target
    self.multiplier = 1000

  def __str__(self):
    return "threadsperblock: {}, blockspergrid: {}".format(self.threadsperblock, self.blockspergrid)


  @staticmethod
  @cuda.jit
  def my_kernel(io_array):
    """ CUDA kernel 
    """
    pos = cuda.grid(1)
    tx = cuda.threadIdx.x 
    if pos < io_array.size:
      io_array[pos] += tx # do the computation


  def run_awhile(self, sec=10):
    start = time.time()
    while time.time() - start < sec:
      self.my_kernel[int(self.multiplier * self.blockspergrid), self.threadsperblock](self._device_data)


  def idle_awhile(self, sec=5):
    time.sleep(sec)
   

  def _boost(self, rate=1.2):
    self.multiplier *= rate


  def _slow_down(self, rate=1.5):
    self.multiplier /= rate
    

  def adjust_speed(self, avg_load):
    if avg_load < self.target * 0.9:
      self._boost()
      print("Adjusted speed: boost")
      return 
    if avg_load > self.target * 1.2:
      self._slow_down()
      print("Adjusted speed: slow_down")
      return 


  @classmethod
  def main(cls, target=50):
    worker = Worker(target)
    print(worker)
    monitor = Monitor()
    monitor.start()
    print("Monitor started: %s" % monitor.is_alive())
    time.sleep(5)
    print("Initial average load", monitor.avg_load)

    while True:
      if monitor.max_load > worker.target * 1.1:
        print("Idle for 5s with load %s" % monitor.max_load)
        worker.idle_awhile(5)
        continue

      print("Run for 10s with load %s and multiplier %s" % (monitor.avg_load, worker.multiplier))
      worker.run_awhile(10)
      worker.adjust_speed(monitor.avg_load)



if __name__ == "__main__":
  import os
  target = float(os.environ.get("TARGET", 50))
  Worker.main(target)
