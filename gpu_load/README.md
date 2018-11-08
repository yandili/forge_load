## 1. 背景需求

最近组内的GPU利用率一直被警告，说是利用率过低。其实GPU这件事和CPU还是有区别的。

第一个问题是内存限制。CPU的话，可以平行的跑很多程序，这样利用率就上去了。但GPU很大程度上受限于内存。如果内存只能装2个进程，再想运行更多的程序也没有办法。

第二个问题是，CPU一般可以通过复制进程来提高利用率，每个进程占用一个CPU核，就可以按任意的比例提高总体利用率。但是GPU的训练任务跑起来的时候，经常一个程序就100%占用了。如果用这种方式占用空闲GPU，别的正常的程序就只能等待了。

不过既然上面要求了，我们也得做。就考虑两个方面的要求，

- 占用尽可能小的内存。
- 控制单进程的GPU资源占用比例。

#### 方案一（废弃）

启动一个接口程序，类似于图像分类任务，模拟用户请求，通过增加请求量的方式来增加负载。

缺点：

- 加载模型的话，会消耗一定比例的内存，我印象比较小的模型也有好几百MB

#### 方案二（采用）

研究一下NVIDIA提供的CUDA接口，直接调用CUDA编写GPU程序，进行简单的并行计算来占用GPU核。这样基本不消耗内存，并且可以精确控制GPU核心数。

## 2. 调研

这一块儿简单了解一下CUDA和python的接口。捡了几个主要概念看了一下。

### CUDA的基本概念

[CUDA的C语言文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

- GPU的核心是Streaming Multiprocessors（sm），数量成千上万。核心有三个概念
- a hierarchy of thread groups, shared memories, and barrier synchronization 
- ![GPU和CPU](http://upload-images.jianshu.io/upload_images/1956647-ceb9d0dc0b8e7447.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- hierachy of thread groups是说，计算任务都是按照矩阵的格式思考。sm的排列可以理解是矩阵，一个子矩阵叫grid，grid的行叫block，具体的元素是thread。总的计算资源占用就是从大的矩阵里规划出来的子矩阵中的所有thread。这个比例基本就对应着GPU的利用率。
- ![Grid](http://upload-images.jianshu.io/upload_images/1956647-90d2c2d4078c9c09.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- shared memories是说，block内部的几个thread，在计算的时候是有一个内部高速cache，如果好几个thread要重复读同一条数据，那最好在算法里把这几个sm放到一个block里。这个我也没仔细看，有个矩阵乘法运算的例子，再补充。
- barrier synchronization是说，block内部的几个thread，是可以等待一起完成的？也没仔细看。

- 规划出来的一个grid，所有的thread是同时拿到一个函数，同时执行。这个函数在CUDA语义下叫kernel。函数里面有变量可以方便每个thread定位自己所在的行数和列数。每个thread通过这个行数和列数，判断自己需要执行的操作。
- 之前有同事告诉我，GPU是一个进程独占全部thread。这个问题现在看来，可能也不一定对，还是看申请了多少thread，剩下的应该不被占用。
- 数据是需要在CPU和GPU之间传递的，有两份存在。

### numba程序实验

python想要调用cuda的功能，需要借助numba。本质上numba是通过预编译python代码加速矩阵运算的。numba提供了一个cuda的接口。[CUDA的python文档](https://numba.pydata.org/numba-doc/dev/cuda/kernels.html)。

比如，下面的python程序，将一个4*128的矩阵，并行填上数字。其中的`grid`, `threadIdx`都是可以定位用。

```python
from numba import cuda
@cuda.jit
def my_kernel(io_array):
    # pos = cuda.grid(1) 是thread个数的一个index, 比如按照后面的配置，2*128=256个
    # tx = cuda.threadIdx.x 是每个block内部0-128的index
    # assert pos == cuda.threadIdx.x + cuda.threadIdx.y * cuda.blockDim.x
    pos = cuda.grid(1)
    tx = cuda.threadIdx.x 
    if pos < io_array.size:
        io_array[pos] += tx # do the computation
```

```python
blocks = 2
threadings = 128
data = np.zeros(512)
# 在运行时指定，用多少thread执行函数，其中的方括号的格式看起来比较奇怪，是对应的在C语言接口里的
#  MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C); 这种
# <<<...>>>
my_kernel[blockspergrid, threadsperblock](data)
```

输出结果如下，

```json
[  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.  11.  12.  13.
  14.  15.  16.  17.  18.  19.  20.  21.  22.  23.  24.  25.  26.  27.
  28.  29.  30.  31.  32.  33.  34.  35.  36.  37.  38.  39.  40.  41.
  42.  43.  44.  45.  46.  47.  48.  49.  50.  51.  52.  53.  54.  55.
  56.  57.  58.  59.  60.  61.  62.  63.  64.  65.  66.  67.  68.  69.
  70.  71.  72.  73.  74.  75.  76.  77.  78.  79.  80.  81.  82.  83.
  84.  85.  86.  87.  88.  89.  90.  91.  92.  93.  94.  95.  96.  97.
  98.  99. 100. 101. 102. 103. 104. 105. 106. 107. 108. 109. 110. 111.
 112. 113. 114. 115. 116. 117. 118. 119. 120. 121. 122. 123. 124. 125.
 126. 127.   0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.  11.
  12.  13.  14.  15.  16.  17.  18.  19.  20.  21.  22.  23.  24.  25.
  26.  27.  28.  29.  30.  31.  32.  33.  34.  35.  36.  37.  38.  39.
  40.  41.  42.  43.  44.  45.  46.  47.  48.  49.  50.  51.  52.  53.
  54.  55.  56.  57.  58.  59.  60.  61.  62.  63.  64.  65.  66.  67.
  68.  69.  70.  71.  72.  73.  74.  75.  76.  77.  78.  79.  80.  81.
  82.  83.  84.  85.  86.  87.  88.  89.  90.  91.  92.  93.  94.  95.
  96.  97.  98.  99. 100. 101. 102. 103. 104. 105. 106. 107. 108. 109.
 110. 111. 112. 113. 114. 115. 116. 117. 118. 119. 120. 121. 122. 123.
 124. 125. 126. 127.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
   0.   0.   0.   0.   0.   0.   0.   0.]
```

可以看到512维的数据只有前两个block被更新了，可以理解逻辑是这样的：

传入一个待写入的内存空间，如果需要写满所有的列，需要`len(blocks) \* len(threads per block)`的thread，总数要超过`data`的行列数。

每个thread，会拿到自己的位置，并判断自己是否执行（比如是否在`data`范围内）。

## 3. 具体实现

### 实现GPU利用率的提升

这个完全可以通过block和thread per block的数量来控制。并且，我们并不需要开一个很大的内存空间，最终thread的数量和data的大小不需要一致。一个thread的kernel即使不做任何事情，也是会被锁定占用的。

### 控制GPU利用率在给定范围

理论上，如果我们知道GPU总共可以提供多少thread数，我们按比例锁定就好，不过没有查到这个总资源量。考虑使用动态的方式调节这个数量。我们使用`Worker`来提升利用率，用`Monitor`来监控利用率。程序启动的时候，设定一个初始的thread数，比如1000。当利用率不足的时候，按固定比例提升thread数；当利用率低于预期时，按固定比例降低thread数。

### 避免占用正常程序资源

`Monitor`检测到负载比较大之后，会自动休眠。只有没有程序负载之后，才启动。

下面是设定占用50%的程序日志，可以看到负载值`load`和thread数的增加比例`multiplier`的变化，

```
threadsperblock: 128, blockspergrid: 4
Monitor started: True
('Initial average load', 0.0)
Run for 10s with load 0.0 and multiplier 1000
Adjusted speed: boost
Run for 10s with load 12.4 and multiplier 1200.0
Adjusted speed: boost
Run for 10s with load 14.4 and multiplier 1440.0
Adjusted speed: boost
Run for 10s with load 15.0 and multiplier 1728.0
Adjusted speed: boost
Run for 10s with load 16.0 and multiplier 2073.6
Adjusted speed: boost
Run for 10s with load 17.6 and multiplier 2488.32
Adjusted speed: boost
Run for 10s with load 24.4 and multiplier 2985.984
Adjusted speed: boost
Idle for 5s with load 69.0
Run for 10s with load 4.8 and multiplier 3583.1808
Adjusted speed: boost
Run for 10s with load 28.8 and multiplier 4299.81696
Adjusted speed: boost
Run for 10s with load 33.6 and multiplier 5159.780352
Adjusted speed: boost
Run for 10s with load 39.6 and multiplier 6191.7364224
Idle for 5s with load 56.00000000000001
Run for 10s with load 0.0 and multiplier 6191.7364224
Run for 10s with load 47.2 and multiplier 6191.7364224
Run for 10s with load 45.6 and multiplier 6191.7364224
Adjusted speed: boost
Run for 10s with load 43.8 and multiplier 7430.08370688
Run for 10s with load 54.8 and multiplier 7430.08370688
Idle for 5s with load 56.00000000000001
Run for 10s with load 8.6 and multiplier 7430.08370688
Idle for 5s with load 56.00000000000001
Run for 10s with load 0.0 and multiplier 7430.08370688
```

最终负载会稳定在50%左右，内存占用为110MB。

![image-20181107204631853.png](https://upload-images.jianshu.io/upload_images/1956647-f37f036a760414e6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


## 4. 部署

考虑到不同机器的配置不同，尤其是cuda和cudnn的版本不同，考虑还是通过docker方式封装。

#### 描述

按照给定比例占用GPU资源

- GPU内存占用为120MB

- 占用比例按照使用情况动态调节，如果有训练程序，会自动让给训练程序使用

- 默认占用50%左右的GPU，闲置时间长的话，会增加到60%

#### 安装需求

机器上需要预先装有cuda，并且下面的命令已经可以输出

```bash
nvidia-docker run -it nvidia/cuda:8.0-cudnn5-devel nvidia-smi
```

#### 编译

按照Dockerfile直接编译

#### 配置参数

```bash
NV_GPU=0  # 显卡号：0或者1
```

#### 启动命令

```bash
NV_GPU={NV_GPU} nvidia-docker run -d\
    --log-driver=json-file --log-opt max-size=3m --log-opt max-file=3 \
    --name forge_load_gpu_{NV_GPU} \
    registry.api.weibo.com/forge_load/forge_load:0.1-gpu
```

#### 终止程序

```bash
docker rm -f forge_load_gpu_${NV_GPU} 
```
