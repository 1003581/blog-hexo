---
title: 虚拟机Hadoop集群安装
date: 2017-09-14 16:01:38
tags: hadoop
categories: 大数据
---

[原文链接](http://dblab.xmu.edu.cn/blog/install-hadoop-simplify/)

<!-- more -->

# 下载软件

- VMWare或者[VirtualBox](https://www.virtualbox.org/wiki/Downloads)
- [Ubuntu](https://mirrors.tuna.tsinghua.edu.cn/ubuntu-releases/)
- [Java](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)
- [Hadoop,选择binary](http://hadoop.apache.org/releases.html)

# 配置

## 系统安装

在虚拟机中安装Ubuntu系统，用户密码均为`linux`，虚拟机名称为`master`

然后开始利用虚拟机的克隆功能复制出`slave1`和`slave2`，注意选择完整克隆。

## 安装SSH、配置SSH无密码登录

若未设置root密码，使用`sudo passwd root`设置root密码

使用`su`切换到root用户

依次记录三台虚拟机的IP地址

**分别**编辑三台虚拟机的network文件，`vim /etc/sysconfig/network`，分别添加

`HOSTNAME=Master`，`HOSTNAME=Slave1`,`HOSTNAME=Slave2`

**分别**打开三台虚拟机的hosts文件，`vim /etc/hosts`，添加如下内容

```
192.168.216.131 Master
192.168.216.130 Slave1
192.168.216.132 Slave2
```

安装并启动SSH

```
apt-get install openssh-server
service ssh start
```

**分别**在三台机器上配置SSH无密码登陆。

```
cd ~/.ssh/                          #若没有该目录，请先执行一次ssh localhost
ssh-keygen -t rsa                   #会有提示，全部回车
ssh-copy-id -i id_rsa.pub root@Master
ssh-copy-id -i id_rsa.pub root@Slave1
ssh-copy-id -i id_rsa.pub root@Slave2
```

此时三台机器可以实现SSH免密钥互通。

## 安装Java

将下载的jdk压缩包复制到Master中，Master中执行如下命令：

```
mv jdk-*-linux-x64.tar.gz /usr/local
cd /usr/local
tar -zxvf jdk-*-linux-x64.tar.gz
rm jdk-*-linux-x64.tar.gz
mv jdk1.8.* jdk
```

复制到Slave1和Slave2

```
scp -r jdk root@Slave1:/usr/local
scp -r jdk root@Slave2:/usr/local
```

**分别**设置三台机器环境变量，命令行输入：

```
echo "export JAVA_HOME=/usr/local/jdk" >> /etc/profile
echo "export PATH=\$JAVA_HOME/bin:\$PATH" >> /etc/profile
echo "export CLASSPATH=.:\$JAVA_HOME/lib/dt.jar:\$JAVA_HOME/lib/tools.jar" >> /etc/profile
source /etc/profile
java -version
```

正确输出java版本即安装成功。

## 安装Hadoop

将下载的hadoop压缩包复制到Master中，Master中执行如下命令：

```
mv hadoop-*.tar.gz /usr/local
cd /usr/local
tar -zxvf hadoop-2.8.0.tar.gz
rm hadoop-*.tar.gz
mv hadoop-* hadoop
mkdir hadoop/tmp
mkdir hadoop/hdfs
mkdir hadoop/hdfs/name
mkdir hadoop/hdfs/data
mkdir hadoop/hdfs/tmp
```

### hadoop-env.sh

```
vim hadoop/etc/hadoop/hadoop-env.sh
```

添加

```
export JAVA_HOME=/usr/jdk
```

### core-site.xml

```
vim hadoop/etc/hadoop/core-site.xml
```

添加

```
<configuration>
    <property>
        <name>hadoop.tmp.dir</name>
        <value>/usr/local/hadoop/tmp</value>
        <final>true</final>
        <description>A base for other temporary directories.</description>
    </property>
    <property>
        <name>fs.default.name</name>
        <value>hdfs://Master:9000</value> <!-- hdfs://Master.Hadoop:22-->  
        <final>true</final>   
    </property>
    <property>
         <name>io.file.buffer.size</name>    
         <value>131072</value>    
    </property>  
</configuration>   
```

### hdfs-site.xml

```
vim hadoop/etc/hadoop/hdfs-site.xml
```

添加

```
<configuration>  
    <property>   
        <name>dfs.replication</name>   
        <value>2</value>   
    </property>   
    <property>   
        <name>dfs.name.dir</name>   
        <value>/usr/local/hadoop/hdfs/name</value>   
    </property>   
    <property>   
        <name>dfs.data.dir</name>   
        <value>/usr/local/hadoop/hdfs/data</value>   
    </property>   
    <property>    
         <name>dfs.namenode.secondary.http-address</name>    
         <value>Master:9001</value>    
    </property>    
    <property>    
         <name>dfs.webhdfs.enabled</name>    
         <value>true</value>    
    </property>    
    <property>    
         <name>dfs.permissions</name>    
         <value>false</value>    
    </property>    
</configuration> 
```

### mapred-site.xml

```
vim hadoop/etc/hadoop/mapred-site.xml
```

修改

```
<configuration>  
    <property>    
          <name>mapreduce.framework.name</name>    
          <value>yarn</value>    
    </property>     
</configuration>
```

### yarn-site.xml

```
vim hadoop/etc/hadoop/yarn-site.xml
```

添加

```
<configuration>  
    <!-- Site specific YARN configuration properties -->  
    <property>    
      <name>yarn.resourcemanager.address</name>    
      <value>Master:18040</value>    
    </property>    
    <property>    
      <name>yarn.resourcemanager.scheduler.address</name>    
      <value>Master:18030</value>    
    </property>    
    <property>    
      <name>yarn.resourcemanager.webapp.address</name>    
      <value>Master:18088</value>    
    </property>    
    <property>    
      <name>yarn.resourcemanager.resource-tracker.address</name>    
      <value>Master:18025</value>    
    </property>    
    <property>    
      <name>yarn.resourcemanager.admin.address</name>    
      <value>Master:18141</value>    
    </property>    
    <property>    
      <name>yarn.nodemanager.aux-services</name>    
      <value>mapreduce_shuffle</value>    
    </property>    
    <property>    
      <name>yarn.nodemanager.aux-services.mapreduce.shuffle.class</name>    
      <value>org.apache.hadoop.mapred.ShuffleHandler</value>    
    </property>    
</configuration>  
```

### masters

```
vim hadoop/etc/hadoop/masters
```

修改

```
Master
```

### slaves

```
vim hadoop/etc/hadoop/slaves
```

修改

```
Slave1
Slave2
```

### 复制到Slave1和Slave2

```
scp -r hadoop root@Slave1:/usr/local
scp -r hadoop root@Slave2:/usr/local
```

### 分别设置环境变量

```
echo "export HADOOP_HOME=/usr/local/hadoop" >> /etc/profile
echo "export PATH=\$HADOOP_HOME/bin:\$HADOOP_HOME/sbin:\$PATH" >> /etc/profile
source /etc/profile
hadoop version
```

# 防火墙

**分别**关闭防火墙

```
firewall-cmd --state
systemctl stop firewalld.service
systemctl disable firewalld.service
firewall-cmd --state
```

# 启动

```
hadoop namenode -format
/usr/local/hadoop/sbin/start-all.sh
```

分别输入`jps`可以看到：

Master

```
9280 NameNode
9665 ResourceManager
9497 SecondaryNameNode
9738 Jps
```

Slave

```
12048 NodeManager
11911 DataNode
12748 Jps
```

# 命令

`hadoop dfsadmin -report`查看集群状态

`http://192.168.216.131:50070`Web查看集群状态

# 测试

## 统计词频

shell中输入命令如下：

```
cd /home
mkdir word_count
cd word_count
vim WordCount.java
```

输入内容如下

```java
import java.io.IOException;
import java.util.StringTokenizer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class WordCount{

    public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable>{
        
        private final static IntWritable one = new IntWritable(1);
        
        private Text word = new Text();
        
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException{
            StringTokenizer itr = new StringTokenizer(value.toString());
            while(itr.hasMoreTokens()){
                word.set(itr.nextToken());
                context.write(word,one);
            }
        }
    }
    
    public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable>{
        
        private IntWritable result = new IntWritable();
        
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum=0;
            for (IntWritable val:values){
                sum += val.get();
            }
            result.set(sum);
            context.write(key,result);
        }
    }
    
    public static void main(String[] args) throws Exception{
        Configuration conf = new Configuration();
        
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        
        if (otherArgs.length < 2){
            System.err.println("Usage: wordcount <int><out>");
            System.exit(2);
        }
        
        Job job = new Job(conf, "word count");
        
        job.setJarByClass(WordCount.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        
        FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
        FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
        
        job.waitForCompletion(true);
    }
    
}
```

进行编译

```
javac -d word_count_class/ WordCount.java
```

打包

```
jar -cvf word_count_class/wordcount.jar *.class
```

测试文件1-file1

```
mkdir input
vim input/file1
```

```
hello world
hello hadoop
hadoop file system
hadoop java api
hello java
```

测试文件2-file2

```
vim input/file2
```

```
new file
hadoop file
hadoop new world
hadoop free home
hadoop free school
strong guan
```

提交

```
hadoop fs -mkdir input_wordcount
hadoop fs -put input/* input_wordcount/
hadoop jar word_count_class/wordcount.jar WordCount input_wordcount output_wordcount
```

查看结果

```
hadoop fs -ls output_wordcount
hadoop fs -cat output_wordcount/part-r-00000
```

## 数据排序

测试文件

```
mkdir sort
mkdir input
vim input/file1
```

```
2
32
654
32
15
756
65223
```

```
vim input/file2
```

```
5956
22
650
92
```

```
vim input/file3
```

```
26
54
6
```

代码

```
vim Sort.java
```

```java
import java.io.IOException;
import java.util.StringTokenizer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class Sort{
    
    //map将输入中的value转化为IntWritable类型，作为输出的key
    public static class Map extends Mapper<Object, Text, IntWritable, IntWritable>{
        
        private static IntWritable data = new IntWritable();
        
        //实现map函数
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException{
            String line = value.toString();
            data.set(Integer.parseInt(line));
            context.write(data, new IntWritable(1));
        }
    }
    
    //reduce将输入中的key复制到输出数据的key上，然后根据输入的value-list中的元素的个数决定key的输出次数
    //用全局的linenum来代表key的位数
    public static class Reduce extends Reducer<IntWritable, IntWritable, IntWritable, IntWritable>{
        
        private static IntWritable linenum = new IntWritable(1);
        
        //实现reduce函数
        public void reduce(IntWritable key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException{
            for (IntWritable val:values){
                context.write(linenum, key);
                
                linenum = new IntWritable(linenum.get()+1);
            }
        }
    }
    
    public static class Partition extends Partitioner<IntWritable, IntWritable>{
        
        @Override
        public int getPartition(IntWritable key, IntWritable value, int numPartitions){
            int MaxNumber = 65223;
            int bound = MaxNumber / numPartitions + 1;
            int keynumber = key.get();
            for(int i=0; i<numPartitions; i++){
                if (keynumber < bound * i && keynumber >= bound * (i-1))
                    return i-1;
            }
            return 0;
        }
    }
    
    public static void main(String[] args) throws Exception{
        Configuration conf = new Configuration();
        
        String[] ioArgs = new String[]{"sort_in","sort_out"};
        String[] otherArgs = new GenericOptionsParser(conf, ioArgs).getRemainingArgs();
        
        if (otherArgs.length != 2){
            System.err.println("Usage: Data Sort <int><out>");
            System.exit(2);
        }
        
        Job job = new Job(conf, "Data Sort");
        
        job.setJarByClass(Sort.class);
        job.setMapperClass(Map.class);
        job.setCombinerClass(Reduce.class);
        job.setReducerClass(Reduce.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        
        FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
        FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
        
        job.waitForCompletion(true);
    }
}
```

编译

```
javac -d data_sort_class/ Sort.java
```

打包

```
jar -cvf datasort.jar *.class
```

提交

```
hadoop fs -mkdir input_datasort
hadoop fs -put input/* input_datasort/
hadoop jar data_sort_class/datasort.jar DataSort input_datasort output_datasort
```

查看结果

```
hadoop fs -ls output_wordcount
hadoop fs -cat output_wordcount/part-r-00000
```
