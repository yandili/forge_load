FROM ubuntu:18.04

MAINTAINER Boris LI <liyandi007@163.com>

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get clean && apt-get update && apt-get install -y \
  curl \
  python-dev\
  python-pip\
  locales \
  rsync \
  wget \
  tzdata \
  iputils-ping \
  && apt-get clean && \
  rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
  -i http://pypi.douban.com/simple 
RUN pip install --upgrade psutil setproctitle \
  -i http://pypi.douban.com/simple --trusted-host pypi.douban.com

## SET LANGUAGE ENCODING
RUN locale-gen zh_CN.UTF-8 en_US.UTF-8 zh_TW zh_TW.UTF-8
ENV LC_CTYPE=zh_CN.UTF-8
ENV LC.MESSAGES=zh_CN.UTF-8
ENV LC_TIME=en_US.UTF-8
ENV PYTHONIOENCODING="utf-8"

## SET TIMEZONE
RUN ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && dpkg-reconfigure -f noninteractive tzdata

## ALIAS
RUN alias rm='rm -i' && alias ps='ps u' && alias tailf='tail -f' && alias grep='grep --color' && alias fgrep='grep -F --color'

## CREATE PROJECT
ARG project_name=cpu_load
ARG project_dir=/home/
RUN mkdir -p ${project_dir}

COPY *py ${project_dir}/${project_name}/

WORKDIR "${project_dir}/${project_name}" 

CMD ["python", "-u", "main.py"]
