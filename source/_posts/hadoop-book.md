---
title: Hadoop权威指南学习
date: 2017-09-14 16:01:16
tags: hadoop
categories: bigdata
---

# 《Hadoop权威指南》

<!-- more -->

[图书官网](http://hadoopbook.com/)

[Github](https://github.com/tomwhite/hadoop-book)

[第三版Github](https://github.com/tomwhite/hadoop-book/tree/3e)

## Ch02 关于MapReduce

[气象数据Sample(已经按年拼成单独文件)](https://github.com/tomwhite/hadoop-book/tree/master/input/ncdc/all)

[原始数据ftp](ftp://ftp.ncdc.noaa.gov/pub/data/noaa/)

[原始数据http](https://www1.ncdc.noaa.gov/pub/data/noaa/)

### 2.2

```
cd ch02/2.2
sh 2-2.max_tempperature.sh
```

### 2.3

```
cd ch02/2.3
sh run.sh
jar -cvf MaxTemperature.jar ./*.class
jar -tvf MaxTemperature.jar
hadoop fs -ls /user/liqiang
hadoop fs -rm -r /user/liqiang/ncdc
hadoop fs -mkdir -p /user/liqiang/ncdc/input
hadoop fs -put ../data/* /user/liqiang/ncdc/input
在管理平台上添加当前用户的动态资源池
hadoop jar MaxTemperature.jar MaxTemperature /user/liqiang/ncdc/input /user/liqiang/ncdc/output
hadoop fs -ls /user/liqiang/ncdc/output
hadoop fs -cat /user/liqiang/ncdc/output/part-r-00000
```


