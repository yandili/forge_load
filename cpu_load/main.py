# -*- coding: utf-8 -*-
#!/usr/bin/python                        
##################################################
# AUTHOR : Yandi LI
# CREATED_AT : 2018-11-01
# LAST_MODIFIED : 2018-11-12 15:46:55
# USAGE : python -u main.py
# PURPOSE : GPU占用程序
##################################################
from __future__ import division
import random
import threading
import multiprocessing
import time
from collections import deque
import psutil

CPU_COUNT = psutil.cpu_count()

class Monitor(threading.Thread):
  """ 后台检测当前GPU占用率
  """

  def __init__(self):
    super(Monitor, self).__init__()
    self.setDaemon(True)
    self._queue = deque([0] * 10, 10)
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
      time.sleep(0.3)

  @staticmethod
  def get_current_load():
    return psutil.cpu_percent()


class Worker(multiprocessing.Process):
  """ CPU占用程序
  - 根据目标target，自动调整需要用到的CPU核心数量
  - 如果monitor检测有其他程序争抢CPU，峰值超过阈值，则自动切断运行
  """

  def __init__(self, target=50):
    super(Worker, self).__init__()
    self.target = target
    self.multiplier = 1
    self.daemon = True


  @staticmethod
  def my_kernel(target, multiplier):
    """ CPU kernel 
    """
    rand = 100 * random.random()
    if rand < target * multiplier:
      start = time.time()
      while time.time() - start < 0.01:
        rand ** 3
    else:
      time.sleep(0.01)


  def run_awhile(self, sec=10):
    start = time.time()
    while time.time() - start < sec:
      self.my_kernel(self.target, self.multiplier)


  def idle_awhile(self, sec=5):
    time.sleep(sec)
   

  def _boost(self, rate=1.05):
    self.multiplier *= rate


  def _slow_down(self, rate=1.1):
    self.multiplier /= rate
    

  def adjust_speed(self, avg_load):
    if avg_load < self.target * 0.8:
      self._boost()
      # print("Adjusted speed: boost")
      return 
    if avg_load > self.target * 1.05:
      self._slow_down()
      # print("Adjusted speed: slow_down")
      return 


  def run(self):
    monitor = Monitor()
    monitor.start()
    print("Monitor started: %s" % monitor.is_alive())
    time.sleep(5)
    print("Initial average load", monitor.avg_load)
    while True:
      if monitor.max_load > self.target * 1.1:
        sec = random.random() * 3 + 1
        # print("Idle for %ss with max_load %s, avg_load %s" % (sec, monitor.max_load, monitor.avg_load))
        self.idle_awhile(sec)
        continue

      sec = random.random() * 3 + 1
      # print("Run for %ss with avg_load %s and multiplier %s" % (sec, monitor.avg_load, self.multiplier))
      self.run_awhile(sec)
      self.adjust_speed(monitor.avg_load)



if __name__ == "__main__":
  import os
  import setproctitle
  setproctitle.setproctitle("forge_load_cpu")
  target = float(os.environ.get("TARGET", 55))
  workers = []
  for i in range(CPU_COUNT):
    worker = Worker(target)
    worker.start()
    print("Worker %d started: %s" % (i, worker.is_alive()))
    workers.append(worker)
  for worker in workers:
    worker.join()

