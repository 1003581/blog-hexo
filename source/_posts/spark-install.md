---
title: Spark集群以及Docker安装
date: 2017-09-14 16:04:02
tags: spark
categories: bigdata
---

相关代码

<!-- more -->

```
docker run -ti -d --name spark ubuntu:zte-16 bash

docker cp jdk-8u131-linux-x64.tar.gz spark:/usr
docker cp scala-2.13.0-M1.tgz spark:/usr
docker cp spark-2.2.0-bin-hadoop2.7.tgz spark:/usr
docker cp zookeeper-3.4.10.tar.gz spark:/usr
docker cp hadoop-2.8.0.tar.gz spark:/usr
docker cp get-pip.py spark:/usr

docker cp core-site.xml spark:/usr
docker cp hdfs-site.xml spark:/usr
docker cp yarn-site.xml spark:/usr
docker cp slaves spark:/usr

docker exec -ti spark bash

apt-get update && apt-get install -y openssh-server
ssh-keygen -q -f "/root/.ssh/id_rsa" -t rsa -P ""
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

cd /usr

tar -zxf jdk-8u131-linux-x64.tar.gz
mv jdk1.8.0_131 jdk
echo "export JAVA_HOME=/usr/jdk" >> /etc/profile
echo "export PATH=\$JAVA_HOME/bin:\$PATH" >> /etc/profile
echo "export CLASSPATH=.:\$JAVA_HOME/lib/dt.jar:\$JAVA_HOME/lib/tools.jar" >> /etc/profile
source /etc/profile
java -version
rm jdk-8u131-linux-x64.tar.gz

tar -zxf scala-2.13.0-M1.tgz
mv scala-2.13.0-M1 scala
echo "export SCALA_HOME=/usr/scala" >> /etc/profile
echo "export PATH=\$SCALA_HOME/bin:\$PATH" >> /etc/profile
source /etc/profile
scala -version
rm scala-2.13.0-M1.tgz

tar -zxf zookeeper-3.4.10.tar.gz
mv zookeeper-3.4.10 zookeeper
echo "export ZOOKEEPER_HOME=/usr/zookeeper" >> /etc/profile
echo "export PATH=\$ZOOKEEPER_HOME/bin:\$PATH" >> /etc/profile
source /etc/profile
rm zookeeper-3.4.10.tar.gz
cp /usr/zookeeper/conf/zoo_sample.cfg /usr/zookeeper/conf/zoo.cfg
sed -i 's/dataDir=\/tmp\/zookeeper/dataDir=\/usr\/zookeepertmp/g' /usr/zookeeper/conf/zoo.cfg
echo "server.1=cloud1:2888:3888" >> /usr/zookeeper/conf/zoo.cfg
echo "server.2=cloud2:2888:3888" >> /usr/zookeeper/conf/zoo.cfg
echo "server.3=cloud3:2888:3888" >> /usr/zookeeper/conf/zoo.cfg
mkdir /usr/zookeepertmp
touch /usr/zookeepertmp/myid
echo 1 > /usr/zookeepertmp/myid

tar -zxf hadoop-2.8.0.tar.gz
mv hadoop-2.8.0 hadoop
echo "export HADOOP_HOME=/usr/hadoop" >> /etc/profile
echo "export PATH=\$HADOOP_HOME/bin:\$HADOOP_HOME/sbin:\$PATH" >> /etc/profile
source /etc/profile
rm hadoop-2.8.0.tar.gz
hadoop version

mkdir /usr/hadooptmp
mv /usr/core-site.xml /usr/hadoop/etc/hadoop/core-site.xml
mv /usr/hdfs-site.xml /usr/hadoop/etc/hadoop/hdfs-site.xml
mv /usr/yarn-site.xml /usr/hadoop/etc/hadoop/yarn-site.xml
mv /usr/slaves /usr/hadoop/etc/hadoop/slaves

tar -zxf spark-2.2.0-bin-hadoop2.7.tgz
rm spark-2.2.0-bin-hadoop2.7.tgz
mv spark-2.2.0-bin-hadoop2.7 spark
echo "export SPARK_HOME=/usr/spark" >> /etc/profile
echo "export PATH=\$SPARK_HOME/bin:\$SPARK_HOME/sbin:\$PATH" >> /etc/profile
source /etc/profile
cp /usr/spark/conf/spark-env.sh.template /usr/spark/conf/spark-env.sh
echo "export SPARK_MASTER_IP=cloud1" >> /usr/spark/conf/spark-env.sh
echo "export SPARK_WORKER_MEMORY=128m" >> /usr/spark/conf/spark-env.sh
echo "export JAVA_HOME=/usr/jdk" >> /usr/spark/conf/spark-env.sh
echo "export SCALA_HOME=/usr/scala" >> /usr/spark/conf/spark-env.sh
echo "export SPARK_HOME=/usr/spark" >> /usr/spark/conf/spark-env.sh
echo "export HADOOP_CONF_DIR=/usr/hadoop/etc/hadoop" >> /usr/spark/conf/spark-env.sh
echo "export SPARK_LIBRARY_PATH=\$SPARK_HOME/lib" >> /usr/spark/conf/spark-env.sh
echo "export SCALA_LIBRARY_PATH=\$SPARK_LIBRARY_PATH" >> /usr/spark/conf/spark-env.sh
echo "export SPARK_WORKER_CORES=1" >> /usr/spark/conf/spark-env.sh
echo "export SPARK_WORKER_INSTANCES=1" >> /usr/spark/conf/spark-env.sh
echo "export SPARK_MASTER_PORT=7077" >> /usr/spark/conf/spark-env.sh
echo "cloud1" >> /usr/spark/conf/slaves
echo "cloud2" >> /usr/spark/conf/slaves
echo "cloud3" >> /usr/spark/conf/slaves

apt-get update && apt-get install -y python
python /usr/get-pip.py
rm /usr/get-pip.py

docker commit spark spark:latest
```

```
docker run -ti -d --name cloud1 -h cloud1 --add-host cloud1:172.17.0.2 --add-host cloud2:172.17.0.3 --add-host cloud3:172.17.0.4 spark:latest bash
docker run -ti -d --name cloud2 -h cloud2 --add-host cloud1:172.17.0.2 --add-host cloud2:172.17.0.3 --add-host cloud3:172.17.0.4 spark:latest bash
docker run -ti -d --name cloud3 -h cloud3 --add-host cloud1:172.17.0.2 --add-host cloud2:172.17.0.3 --add-host cloud3:172.17.0.4 spark:latest bash

docker exec -ti cloud1 bash

docker exec -ti cloud2 bash
echo 2 > /usr/zookeepertmp/myid
exit
docker exec -ti cloud3 bash
echo 3 > /usr/zookeepertmp/myid
exit

docker exec cloud1 /usr/zookeeper/bin/zkServer.sh start
docker exec cloud2 /usr/zookeeper/bin/zkServer.sh start
docker exec cloud3 /usr/zookeeper/bin/zkServer.sh start
```










python update
```
tar -xf Python-2.7.13.tgz
rm Python-2.7.13.tgz
echo "export PYTHONHOME=/usr/local/python2" >> /etc/profile
source /etc/profile
apt-get update
apt-get install -y gcc make zlib* libbz2-dev libgdbm-dev liblzma-dev libreadline-dev libsqlite3-dev libssl-dev tcl-dev tk-dev dpkg-dev
pushd Python-2.7.13
./configure --prefix="${PYTHONHOME}" --enable-unicode=ucs4
sed -i 's/#zlib/zlib/g' Modules/Setup
make
make install
popd
rm -rf Python-2.7.12
python get-pip.py
rm get-pip.py
```

安装SSH

修改允许root登录

配置免密钥登录

```
在三台机器上分别执行,每条指令单独执行
ssh-keygen -t rsa
ssh-copy-id -i ~/.ssh/id_rsa.pub root@10.40.64.205
ssh-copy-id -i ~/.ssh/id_rsa.pub root@10.40.64.206
ssh-copy-id -i ~/.ssh/id_rsa.pub root@10.42.10.61
```

分别配置hostname

配置host

```
分别执行
vim /etc/hosts

10.42.10.61 Master
10.40.64.205 Slave1
10.40.64.206 Slave2
```

安装Java

```
cd /usr/local
scp root@10.42.10.61:/root/spark/jdk-8u131-linux-x64.tar.gz ./
tar -zxf jdk-8u131-linux-x64.tar.gz
rm jdk-8u131-linux-x64.tar.gz
echo "export JAVA_HOME=/usr/local/jdk1.8.0_131" >> /etc/profile
echo "export PATH=\$JAVA_HOME/bin:\$PATH" >> /etc/profile
echo "export CLASSPATH=.:\$JAVA_HOME/lib/dt.jar:\$JAVA_HOME/lib/tools.jar" >> /etc/profile
source /etc/profile
apt-get remove -y openjdk*  
source /etc/profile
java -version
```

安装hadoop

```
cd /usr/local
scp root@10.42.10.61:/root/spark/hadoop-2.8.0.tar.gz ./
tar -zxf hadoop-2.8.0.tar.gz
rm hadoop-2.8.0.tar.gz
echo "export HADOOP_HOME=/usr/local/hadoop-2.8.0" >> /etc/profile
echo "export PATH=\$HADOOP_HOME/bin:\$HADOOP_HOME/sbin:\$PATH" >> /etc/profile
source /etc/profile
hadoop version
```

修改配置

hadoop守护进程的运行环境配置

```
vim ${HADOOP_HOME}/etc/hadoop/hadoop-env.sh
```
```
export JAVA_HOME=/usr/local/jdk1.8.0_131
```

core-site.xml:

```
mkdir /usr/local/hadoop-2.8.0/tmp
vim ${HADOOP_HOME}/etc/hadoop/core-site.xml
```
```
<configuration>
    <property>
    <!-- 指定hdfs的namenode为Master -->
    　　<name>fs.defaultFS</name>
    　　<value>hdfs://Master:9000</value>
    </property>
    <!-- Size of read/write buffer used in SequenceFiles. -->
    <property>
     　　<name>io.file.buffer.size</name>
     　　<value>131072</value>
   </property>
    <!-- 指定hadoop临时目录,自行创建 -->
    <property>
        <name>hadoop.tmp.dir</name>
        <value>/usr/local/hadoop-2.8.0/tmp</value>
    </property>
</configuration>
```

hdfs-site.xml:配置namenode和datanode存储命名空间和log的路径

```
mkdir /usr/local/hadoop-2.8.0/hdfs
mkdir /usr/local/hadoop-2.8.0/hdfs/name
mkdir /usr/local/hadoop-2.8.0/hdfs/data
vim ${HADOOP_HOME}/etc/hadoop/hdfs-site.xml
```
```
<configuration>
    <!-- 备份数：默认为3-->
     <property>
        <name>dfs.replication</name>
         <value>2</value>
     </property>
    <!-- namenode-->
     <property>
         <name>dfs.namenode.name.dir</name>
         <value>file:/usr/local/hadoop-2.8.0/hdfs/name</value>
     </property>
    <!-- datanode-->
     <property>
         <name>dfs.datanode.data.dir</name>
         <value>file:/usr/local/hadoop-2.8.0/hdfs/data</value>
     </property>
    <!--权限控制：false：不做控制即开放给他用户访问 -->
     <property>
         <name>dfs.permissions</name>
         <value>false</value>
     </property>
</configuration>
```

mapred-site.xml：配置MapReduce。

```
vim ${HADOOP_HOME}/etc/hadoop/mapred-site.xml
```
```
<configuration>
    <!-- mapreduce任务执行框架为yarn-->
    <property>
         <name>mapreduce.framework.name</name>
         <value>yarn</value>
    </property>
    <!-- mapreduce任务记录访问地址-->
    <property>
          <name>mapreduce.jobhistory.address</name>
          <value>Master:10020</value>
     </property>
    <property>                 
           <name>mapreduce.jobhistory.webapp.address</name>
            <value>Master:19888</value>
     </property>
</configuration>
```

yarn-site.xml：配置resourcesmanager和nodemanager

```
vim ${HADOOP_HOME}/etc/hadoop/yarn-site.xml
```
```
<configuration>
    <property>
        <description>The hostname of the RM.</description>
        <name>yarn.resourcemanager.address</name>
        <value>Master:8032</value>
    </property>
    <property>
        <name>yarn.resourcemanager.scheduler.address</name>
        <value>Master:8030</value>
    </property>
    <property>
        <name>yarn.resourcemanager.resource-tracker.address</name>
        <value>Master:8031</value>
    </property>
    <property>                 
        <name>yarn.resourcemanager.admin.address</name>
        <value>Master:8033</value>
    </property>
    <property>
        <name>yarn.resourcemanager.webapp.address</name>
        <value>Master:8088</value>
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

slaves:配置集群的DataNode节点,这些节点是slaves,NameNode是Master。在conf/slaves文件中列出所有slave的主机名或者IP地址，一行一个。配置如下：

```
vim ${HADOOP_HOME}/etc/hadoop/slaves
```
```
Slave1
Slave2
```


启动hadoop

Master

```
hdfs namenode -format
start-all.sh
```

启动正确后

```
Master显示
18595 ResourceManager
18036 NameNode
18340 SecondaryNameNode
Slave显示
12964 DataNode
13237 Jps
13114 NodeManage
```

查看存活的端口

```
http://10.42.10.61:8088/
```

```
http://10.42.10.61:50070
```


安装Scala Spark

```
cd /usr/local
scp root@10.42.10.61:/root/spark/scala-2.13.0-M1.tgz ./
scp root@10.42.10.61:/root/spark/spark-2.2.0-bin-hadoop2.7.tgz ./
tar -zxf scala-2.13.0-M1.tgz
tar -zxf spark-2.2.0-bin-hadoop2.7.tgz
rm scala-2.13.0-M1.tgz
rm spark-2.2.0-bin-hadoop2.7.tgz
echo "export SCALA_HOME=/usr/local/scala-2.13.0-M1" >> /etc/profile
echo "export PATH=\$SCALA_HOME/bin:\$PATH" >> /etc/profile
source /etc/profile
scala -version
echo "export SPARK_HOME=/usr/local/spark-2.2.0-bin-hadoop2.7" >> /etc/profile
echo "export PATH=\$SPARK_HOME/bin:\$SPARK_HOME/sbin:\$PATH" >> /etc/profile
source /etc/profile
```

配置修改

spark-env.sh:spark执行任务的环境配置，需要根据自己的机器配置来设置，内存和核心数配置的时候主要不要超出虚拟机的配置，尤其是存在默认值的配置需要仔细查看，修改。

```
cp ${SPARK_HOME}/conf/spark-env.sh.template ${SPARK_HOME}/conf/spark-env.sh
vim ${SPARK_HOME}/conf/spark-env.sh
```
```
export SPARK_DIST_CLASSPATH=$(/usr/local/hadoop-2.8.0/bin/hadoop classpath)
#Master----config
SPARK_LOCAL_DIRS=/usr/local/spark-2.2.0-bin-hadoop2.7/local #配置spark的local目录
SPARK_MASTER_IP=Master #master节点ip或hostname
SPARK_MASTER_WEBUI_PORT=8085 #web页面端口

#export SPARK_MASTER_OPTS="-Dspark.deploy.defaultCores=4" #spark-shell启动使用核数
SPARK_WORKER_CORES=1 #Worker的cpu核数
SPARK_WORKER_MEMORY=1g #worker内存大小
SPARK_WORKER_DIR=/usr/local/spark-2.2.0-bin-hadoop2.7/worker #worker目录
SPARK_WORKER_OPTS="-Dspark.worker.cleanup.enabled=true -Dspark.worker.cleanup.appDataTtl=604800" #worker自动清理及清理时间间隔
SPARK_HISTORY_OPTS="-Dspark.history.ui.port=18080 -Dspark.history.retainedApplications=3 -Dspark.history.fs.logDirectory=hdfs://Master:9000/spark/history" #history server页面端口>、备份数、log日志在HDFS的位置
SPARK_LOG_DIR=/usr/local/spark-2.2.0-bin-hadoop2.7/logs #配置Spark的log日志
JAVA_HOME=/usr/local/jdk1.8.0_131 #配置java路径
SCALA_HOME=/usr/local/scala-2.13.0-M1 #配置scala路径
HADOOP_HOME=/usr/local/hadoop-2.8.0/lib/native #配置hadoop的lib路径
HADOOP_CONF_DIR=/usr/local/hadoop-2.8.0/etc/hadoop/ #配置hadoop的配置路径

export SPARK_LOCAL_IP=127.0.0.1
```

spark-default.conf:

```
cp ${SPARK_HOME}/conf/spark-defaults.conf.template ${SPARK_HOME}/conf/spark-defaults.conf
vim ${SPARK_HOME}/conf/spark-defaults.conf
```
```
spark.master                     spark://Master:7077
spark.eventLog.enabled           true
spark.eventLog.dir               hdfs://Master:9000/spark/history
spark.serializer                 org.apache.spark.serializer.KryoSerializer
spark.driver.memory              1g
spark.executor.extraJavaOptions  -XX:+PrintGCDetails -Dkey=value -Dnumbers="one two three"
```

Slaves

```
cp ${SPARK_HOME}/conf/slaves.template ${SPARK_HOME}/conf/slaves
vim ${SPARK_HOME}/conf/slaves
```
```
Master
Slave1
Slave2
```

```
scp -r ${SPARK_HOME} root@Slave1:/usr/local
scp -r ${SPARK_HOME} root@Slave2:/usr/local
```


```
hadoop fs -mkdir -p /spark/history
start-all.sh
jps

18595 ResourceManager
18036 NameNode
21108 Jps
18340 SecondaryNameNode
20953 Worker
20684 Master

12964 DataNode
13957 Jps
13114 NodeManager
13899 Worker

run-example SparkPi 2>&1 | grep "Pi is roughly"
```

安装新版本Pip

```
apt-get update
apt-get install -y python-pip python-dev
```

拷贝pip-9.0.1-py2.py3-none-any.whl文件

配置公司pip源

```
pip install pip-9.0.1-py2.py3-none-any.whl
```

tensorflow

```
pip2 install tensorflow
pip2 install tensorflowonspark
```

Run

```
rm -rf mnist_model/
rm -rf csv

# save images and labels as CSV files
spark-submit \
--master yarn \
--deploy-mode cluster \
mnist_data_setup.py \
--output csv \
--format csv
```





```
echo "hello world hello Hello" > test.txt
hadoop fs -mkdir /input
hadoop fs -put test.txt /input
hadoop jar /usr/local/hadoop-2.8.0/share/hadoop/mapreduce/hadoop-mapreduce-examples-*.jar wordcount /input /output
```
