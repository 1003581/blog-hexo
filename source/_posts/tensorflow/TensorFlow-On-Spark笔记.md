---
title: TensorFlow On Spark笔记
date: 2017-09-14 16:04:24
tags: 
- tensorflow
- spark
categories: tensorflow
---

### 资料
- [Github](https://github.com/yahoo/TensorFlowOnSpark)
- [Github Wiki](https://github.com/yahoo/TensorFlowOnSpark/wiki)
- [Yahoo开源TensorFlow On Spark](http://www.datagold.com.cn/archives/7020.html)
- [Open Sourcing TensorFlowOnSpark: Distributed Deep Learning on Big-Data Clusters](http://yahoohadoop.tumblr.com/post/157196317141/open-sourcing-tensorflowonspark-distributed-deep)
- [TensorflowOnSpark：1)Standalone集群初体验 - PJ·Javis的专栏 - CSDN博客](http://blog.csdn.net/jiangpeng59/article/details/72867368)
- [如何安装Spark & TensorflowOnSpark - 每天get√新知识 - CSDN博客](http://blog.csdn.net/fishseeker/article/details/61918138)
<!-- more -->
### [Background, Architecture and Interface](http://yahoohadoop.tumblr.com/post/157196317141/open-sourcing-tensorflowonspark-distributed-deep)

### [Start TensorFlowOnSpark on a Standalone Cluster installed locally](https://github.com/yahoo/TensorFlowOnSpark/wiki/GetStarted_standalone)

环境变量整理

```
export TFoS_HOME=/usr/local/TensorFlowOnSpark
export SPARK_HOME=${TFoS_HOME}/scripts/spark-1.6.0-bin-hadoop2.6
export PATH=${SPARK_HOME}/bin:${PATH}
export MASTER=spark://$(hostname):7077
export SPARK_WORKER_INSTANCES=2
export CORES_PER_WORKER=1 
export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES})) 
```

MNIST ZIP转化为csv命令

```
${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
${TFoS_HOME}/examples/mnist/mnist_data_setup.py \
--output examples/mnist/csv \
--format csv
```

训练任务提交

```
${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--py-files ${TFoS_HOME}/examples/mnist/spark/mnist_dist.py \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
${TFoS_HOME}/examples/mnist/spark/mnist_spark.py \
--cluster_size ${SPARK_WORKER_INSTANCES} \
--images examples/mnist/csv/train/images \
--labels examples/mnist/csv/train/labels \
--format csv \
--mode train \
--tensorboard \
--model mnist_model
```

推理任务提交

```
${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--py-files ${TFoS_HOME}/examples/mnist/spark/mnist_dist.py \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
${TFoS_HOME}/examples/mnist/spark/mnist_spark.py \
--cluster_size ${SPARK_WORKER_INSTANCES} \
--images examples/mnist/csv/test/images \
--labels examples/mnist/csv/test/labels \
--mode inference \
--format csv \
--model mnist_model \
--tensorboard \
--output predictions
```

运行Jupyter Notebook时，修改命令为

```
pushd ${TFoS_HOME}/examples/mnist
PYSPARK_DRIVER_PYTHON="jupyter" \
PYSPARK_DRIVER_PYTHON_OPTS="notebook --no-browser --ip=`10.42.10.61` --allow-root" \
pyspark  --master ${MASTER} \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
--py-files ${TFoS_HOME}/examples/mnist/spark/mnist_dist.py \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME"
```

集群运行命令（未整理）

```
spark-submit \
--py-files examples/mnist/spark/mnist_dist.py \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
${TFoS_HOME}/examples/mnist/spark/mnist_spark.py \
--cluster_size ${SPARK_WORKER_INSTANCES} \
--images examples/mnist/csv/train/images \
--labels examples/mnist/csv/train/labels \
--format csv \
--mode train \
--model mnist_model


spark-submit \
--py-files examples/mnist/spark/mnist_dist.py \
--conf spark.cores.max=2 \
--conf spark.task.cpus=1 \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
examples/mnist/spark/mnist_spark.py \
--cluster_size 2 \
--images examples/mnist/csv/train/images \
--labels examples/mnist/csv/train/labels \
--format csv \
--mode train \
--model mnist_model


spark-submit \
--py-files ${TFoS_HOME}/examples/mnist/spark/mnist_dist.py \
--conf spark.cores.max=1 \
--conf spark.task.cpus=3 \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
--conf spark.executorEnv.LD_LIBRARY_PATH="${JAVA_HOME}/jre/lib/amd64/server" \
--conf spark.executorEnv.CLASSPATH="$(hadoop classpath --glob):${CLASSPATH}" \
${TFoS_HOME}/examples/mnist/spark/mnist_spark.py \
--cluster_size 3 \
--images examples/mnist/csv/train/images \
--labels examples/mnist/csv/train/labels \
--format csv \
--mode train \
--model mnist_model

spark-submit \
--py-files ${TFoS_HOME}/examples/mnist/spark/mnist_dist.py \
--conf spark.cores.max=2 \
--conf spark.task.cpus=1 \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
--conf spark.executorEnv.LD_LIBRARY_PATH="${JAVA_HOME}/jre/lib/amd64/server" \
--conf spark.executorEnv.CLASSPATH="$(hadoop classpath --glob):${CLASSPATH}" \
${TFoS_HOME}/examples/mnist/spark/mnist_spark.py \
--cluster_size 2 \
--images examples/mnist/csv/train/images \
--labels examples/mnist/csv/train/labels \
--format csv \
--mode train \
--model mnist_model

zip -r tfspark.zip tensorflowonspark/*

spark-submit \
--conf spark.executorEnv.LD_LIBRARY_PATH="${JAVA_HOME}/jre/lib/amd64/server" \
--conf spark.executorEnv.CLASSPATH="$(hadoop classpath --glob):${CLASSPATH}" \
--py-files ${TFoS_HOME}/examples/mnist/spark/mnist_dist.py,${TFoS_HOME}/tfspark.zip \
--conf spark.cores.max=3 \
--conf spark.task.cpus=1 \
${TFoS_HOME}/examples/mnist/spark/mnist_spark.py \
--cluster_size 3 \
--images examples/mnist/csv/train/images \
--labels examples/mnist/csv/train/labels \
--format csv \
--mode train \
--model mnist_model

spark-submit \
--conf spark.executorEnv.LD_LIBRARY_PATH="${JAVA_HOME}/jre/lib/amd64/server" \
--conf spark.executorEnv.CLASSPATH="$(hadoop classpath --glob):${CLASSPATH}" \
--py-files ${TFoS_HOME}/tfspark.zip,${TFoS_HOME}/examples/mnist/spark/mnist_dist.py \
--conf spark.cores.max=3 \
--conf spark.task.cpus=1 \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
${TFoS_HOME}/examples/mnist/spark/mnist_spark.py \
--cluster_size 3 \
--images examples/mnist/csv/test/images \
--labels examples/mnist/csv/test/labels \
--mode inference \
--format csv \
--model mnist_model \
--output predictions
```
