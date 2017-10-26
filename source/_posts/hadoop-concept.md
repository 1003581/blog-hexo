---
title: Hadoop相关概念
date: 2017-09-14 16:01:29
tags: hadoop
categories: 大数据
---

# 分布式文件系统

<!-- more -->

HDFS，Hadoop Distributed File System

块默认大小为64MB，设计如此大是为了最小化寻址空间。

`% hadoop fsck / -files -blocks`

列出文件系统中各个文件由哪些快构成

## namenode和datanode

namenode为管理者，管理文件系统中的命名空间，维护文件系统树及整棵树内所有的文件和目录。这些信息以2个文件形式永久保存在本地磁盘：命名空间镜像文件、编辑日志文件。namenode也记录每个文件中每个块所在的数据节点信息，但是并不永久保存，因为这些信息会在系统启动时由datanode重建。

datanode是工作者，它根据需要存储并检索数据块（由客户端或者namenode调度），并且定期向namenode发送它们所存储的块的列表。

namenode失效将导致文件系统无法使用，为了对namenode进行容错，一种方法是备份那些组成文件系统元数据持久状态的文件，另一种方法是运行一个辅助namenode，这个namenode不是真正的namenode，它是用来定期通过编辑日志合并命名空间镜像，以防止编辑日志过大，它会在namenode失效时启用，但是难免会丢失部分数据。

## 联邦HDFS

namenode在内存中保存文件系统中每个文件和每个数据块的引用关系，这意味着内存会限制集群的扩展，hadoop2.x版本引入联邦HDFS，该系统允许添加namenode实现拓展，其中每个namenode管理文件系统中的一部分，如一个管理/usr，另一个管理/share。

联邦环境下，每个namenode维护一个命名空间卷，包括命名空间的源数据和在该命名空间下的文件的所有数据块的数据快池。命名空间卷是相互独立的，两两互不通信，一个失效也不会影响另外一个。数据块池不再进行切分，所以集群中的datanode需要注册到每个namenode，并存储来自多个数据块池中的数据块。

## 命令行接口

配置伪分布式时候，属性项fs.default.name设置为hfs://localhost/，意思是使用hdfs作为hadoop的默认文件系统，将在localhost的8020端口上运行namenode。属性项dfs.replication设置为1，hdfs就不会默认将文件系统块复本设置为3，因为我们只有1到2个datanode，无法复制3份，会一直给副本不足的警告。

% hadoop fs -help

获取每个命令的详细帮助

从本地系统中复制一个文件到hdfs

% hadoop fs -copyFromLocal input/docs/quangle.txt hdfs://localhost/user/tom/quangle.txt

如果主机的默认URI在core-site.xml中设置，则输入

% hadoop fs -copyFromLocal input/docs/quangle.txt /user/tom/quangle.txt

使用相对目录也可，默认是home目录

% hadoop fs -copyFromLocal input/docs/quangle.txt quangle.txt

再将文件复制回本地系统，检查是否一致，正确结果为相同

% hadoop fs -copyToLocal quangle.txt quangle.coy.txt

% md5 input/docs/quangle.txt quangle.copy.txt

新建一个目录

% hadoop fs -mkdir books

查看hdfs文件列表

% hadoop fs -ls

输出如下：

Found 2 items

drwxr-xr-x  -   tom supergroup  0   2009-04-02  22:41   /user/tom/books

-rw-r--r--      1   tom supergroup  118 2009-04-02  22:29   /user/tom/quangle.txt

| 文件模式       | 备份数目 | 所属用户 | 所属组别    | 文件大小     | 最后修改日期     | 最后修改时间 | 文件或者目录的队绝对目录                |
| ---------- | ---- | ---- | ------- | -------- | ---------- | ------ | --------------------------- |
| -rw-r----- | 3    | root | liqiang | 13546617 | 2017-06-19 | 10:16  | /user/liqiang/blockinfo.txt |
| drwxr-x--- | -    | root | liqaing | 0        | 2017-06-19 | 10:30  | /user/liqiang/dir           |

依次显示的为文件模式、文件的备份数、文件所属用户、文件所属组别、文件大小、文件最后修改日期、文件最后修改时间、文件或者目录的绝对路径

文件模式中r为只读权限，w为写入权限，x为执行权限。

文件备份数中，目录显示为空，因为目录作为元数据保存在namenode中，而非datanode中。

文件大小以字节为单位，目录为0。

## Hadoop文件系统

Java抽象类org.apache.hadoop.fs.FileSystem定义了Hadoop中的一个文件系统接口。

列出本地文件系统根目录下的文件

% hadoop fs -ls file:///

HTTP

通过HTTP访问HDFS由2种方法，一种是直接访问，由namenode内嵌的web服务器（端口50070）提供目录服务，目录以xml或者json格式存储；由datanode内嵌的web服务器（端口50075）以数据流的形式传输。另一种访问方式是通过HDFS代理，客户端通常使用DistributedFileSystemAPI来访问HDFS。

C

libhdfs库，在发行包的Libhdfs/docs/api目录下寻找CAPI的相关文档

FUSE

用户空间文件系统（Filesystem in Userspace），实现了一种将现有的文件系统整合为一个Unix文件系统，并可以进行挂载，然后利用Unix工具进行交互。

# FileSystem类

通过URL读取数据

从hadoop文件系统中读取文件，使用java.net.URL打开对象流

```java
InputStream in = null;
try{
    in = new URL("hdfs://host/path").openStream();
    // process in
}finally{
    IOUtils.closeStream(in);
}
```

在此之前，需要实例化一个对象。

通过URLStreamHandler实例以标准输出方式显示Hadoop文件系统的文件

```java
public class URLCat {
static {
URL.setURLStreamHandlerFactory(new FsUrlStreamHandlerFactory());
}

public static void main(String[] args) throws Exception{
InputStream in = null;
try {
in = new URL(args[0]).openStream();
IOUtils.copyBytes(in, System.out, 4096, false);
} finally {
IOUtils.closeStream(in);
}
}
}
```

运行
% hadoop URLCat hdfs://localhost/user/tom/quangle.txt

注：
copyBytes函数参数（输入流，输出流，复制的缓冲区大小，复制结束后是否关闭数据流）

本例中finally中结束in，故copyBytes函数最后一个参数为fasle

通过FileSystem API读取数据

第一个方法中需要实例化，若其他程序实例化过，则无法再次实例化，程序会运行失败。

## Hadoop

查看Yarn Web页面 http://ip:50070/

查看Hadoop总览页面 http://ip:8088]

列出所有文件由哪些块组成，由于信息较多，将重定向到文件中去 `hadoop fsck / -files -blocks >> blockinfo.txt`

复制文件到hdfs中`hadoop fs -copyFromLocal blockinfo.txt /user/liqiang/blockinfo.txt`

复制文件到本地`hadoop fs -copyToLocal /user/liqiang/blockinfo.txt blockinfo.txt`

## Hive

给当前用户提升权限`grant all on database test to user root;`

切换数据库`use db_name`

创建表

```
create table records(year STRING, temperature INT , quality INT)
row format delimited
  fields terminated by '\t';
```

导入数据

```
load data local inpath '/root/lq/ch02/data/1901.gz'
overwrite into table records;
```

## Spark

[link](http://www.cnblogs.com/yangzhang-home/p/6056133.html)

```
pyspark
textFile=sc.textFile("file:///opt/ZDH/parcels/lib/spark/README.md")
textFile.count()
textFile.first()
linesWithSpark = textFile.filter(lambda line: "Spark" in line)
textFile.filter(lambda line: "Spark" in line).count()  # 有多好行含有“Spark”这一字符串
textFile.map(lambda line: len(line.split())).reduce(lambda a, b: a if (a>b) else b)
wordCounts = textFile.flatMap(lambda line: line.split()).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a+b)
wordCounts.collect()
linesWithSpark.cache()
linesWithSpark.count()
linesWithSpark.count()
```
