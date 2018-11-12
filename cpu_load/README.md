## 1. 背景需求

组内需要将CPU的空闲利用率维持在给定的水平。

现有的占用程序是从两个方式实现的，

- 占用固定比例的CPU，问题是可能会影响线上性能（频道流出现过一次这样的问题）
- 在非请求时间定时启动占用程序，问题是无法提高平均CPU利用率

为了解决目前的问题，这里重新实现了一个动态调整资源的程序，当线上请求密集时，自动休眠；当线上利用率不足时，通过占用程序补足。

## 2. 具体实现

### 占用给定比例的CPU

逻辑是在给定的时间段内，一定比例时间睡眠，一定比例时间进行计算。代码片段如下，

```python
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

```

### 避免占用正常程序资源

`Monitor`检测到负载比较大之后，会自动休眠。只有没有程序负载之后，才启动。


## 3. 部署

考虑到不同机器的配置不同，考虑通过docker方式封装。

#### 描述

按照给定比例占用CPU资源

- 占用比例按照使用情况动态调节，如果有训练程序，会自动让给训练程序使用

- 默认占用50%左右的GPU，闲置时间长的话，会增加到60%

#### 编译

按照Dockerfile直接编译

#### 启动命令

```bash
docker run -d \
    --name forge_load_cpu \
    registry.api.weibo.com/forge_load/forge_load:0.1-cpu
```

#### 终止程序

```bash
docker rm -f forge_load_cpu 
```

