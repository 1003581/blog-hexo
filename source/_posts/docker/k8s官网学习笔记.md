---
title: k8s官网学习笔记
date: 2017-12-01 15:11:50
tags: k8s
categories: docker
---

k8s [https://kubernetes.io/docs/home/](https://kubernetes.io/docs/home/)
<!-- more -->

# [Kubernetes Documentation](https://kubernetes.io/docs/home/)

文档分为如下几部分

- [Interactive Tutorial](https://kubernetes.io/docs/tutorials/kubernetes-basics/) 在线交互式教程
- [Installing/Setting Up Kubernetes](https://kubernetes.io/docs/setup/pick-right-solution/) 在本地或者云端安装集群
- [Concepts](https://kubernetes.io/docs/concepts/) kubernetes如何运行的深度讲解
- [Tasks](https://kubernetes.io/docs/tasks/) 通用kubernetes任务的详细地介绍
- [Tutorials](https://kubernetes.io/docs/tutorials/) kubernetes工作流程的详细演练
- [API and Command References](https://kubernetes.io/docs/reference/)

# [Tutorials 在线教程方面](https://kubernetes.io/docs/tutorials/)

## [Kubernetes Basics](https://kubernetes.io/docs/tutorials/kubernetes-basics/)

此为文档的第二部分，交互式教程

### [Overview](https://kubernetes.io/docs/tutorials/kubernetes-basics/)

阅读本章节，你可以：

- 在集群上部署一个容器化应用
- 规模化部署
- 使用一个新的软件版本来更新容器化应用
- 调试容器化应用

### 1. Create a Cluster

#### [Using Minikube to Create a Cluster](https://kubernetes.io/docs/tutorials/kubernetes-basics/cluster-intro/)

一个Kubernetes集群包含两种资源：

- Master（管理集群），包含如调度应用程序、维护应用程序所需的状态、扩展应用程序以及推出新的更新。
- Nodes（跑应用），是一个虚拟机或者一个物理机 。每个节点包含一个`Kubelet`（是一个用于管理节点和与Master通信的Agent程序），一个容器管理工具（比如Docker）。

![集群图](http://upload-images.jianshu.io/upload_images/5952841-67dddea93f772715.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

[原图](https://d33wubrfki0l68.cloudfront.net/99d9808dcbf2880a996ed50d308a186b5900cec9/40b94/docs/tutorials/kubernetes-basics/public/images/module_01_cluster.svg)

当用户需要启动一个应用时，用户会通知Master，Master会去调度一个node来运行这个应用。

Master与Nodes之间采用Kubernetes API进行通信，所以允许终端用户直接使用这些API去与集群进行交互。

[Minikube](https://github.com/kubernetes/minikube)是一个轻量级型的Kubernetes本地部署实现，其在本机上创建了一个虚拟机并部署了一个只包含一个节点的简单集群。其提供了集群中的start、stop、status和delete命令。

#### [Interactive Tutorial - Creating a Cluster](https://kubernetes.io/docs/tutorials/kubernetes-basics/cluster-interactive/) 

```shell
$ minikube version
minikube version: v0.17.1-katacoda
$ minikube start
Starting local Kubernetes cluster...
$ kubectl version
Client Version: version.Info{Major:"1", Minor:"8", GitVersion:"v1.8.0", GitCommit:"6e937839ac04a38cac63e6a7a306c5d035fe7b0a", GitTreeState:"clean", BuildDate:"2017-09-28T22:57:57Z", GoVersion:"go1.8.3", Compiler:"gc", Platform:"linux/amd64"}
Server Version: version.Info{Major:"1", Minor:"5", GitVersion:"v1.5.2", GitCommit:"08e099554f3c31f6e6f07b448ab3ed78d0520507", GitTreeState:"clean", BuildDate:"1970-01-01T00:00:00Z", GoVersion:"go1.7.1", Compiler:"gc", Platform:"linux/amd64"}
$ kubectl cluster-info
Kubernetes master is running at http://host01:8080
kubernetes-dashboard is running at http://host01:8080/api/v1/namespaces/kube-system/services/kubernetes-dashboard/proxy
monitoring-grafana is running at http://host01:8080/api/v1/namespaces/kube-system/services/monitoring-grafana/proxy

To further debug and diagnose cluster problems, use 'kubectl cluster-info dump'.
$ kubectl get nodes
NAME      STATUS    ROLES     AGE       VERSION
host01    Ready     <none>    9s        v1.5.2
```

### 2. Deploy an App

#### [Using kubectl to Create a Deployment](https://kubernetes.io/docs/tutorials/kubernetes-basics/deploy-intro/) 

创建一个kubernetes部署配置，这个指示了如何创建和更新应用。每当应用被创建，master都会一直监控这些句柄，如果节点挂掉了，则会安排其他节点去替代它。

kubernetes命令行交互工具**Kubectl**。该工具使用kubernetes API去跟集群进行交互，创建应用时，需要指定镜像地址和repicas副本数目。

#### [Interactive Tutorial - Deploying an App](https://kubernetes.io/docs/tutorials/kubernetes-basics/deploy-interactive/) 

```shell
Kubernetes Bootcamp Terminal
$
$ sleep 1; ~/.bin/launch.sh
Starting Kubernetes...
Kubernetes Started
$ kubectl
kubectl controls the Kubernetes cluster manager.

Find more information at https://github.com/kubernetes/kubernetes.

Basic Commands (Beginner):
  create         Create a resource from a file or from stdin.
  expose         Take a replication controller, service, deployment or pod and expose it as a new
Kubernetes Service
  run            Run a particular image on the cluster
  set            Set specific features on objects
  run-container  Run a particular image on the cluster. This command is deprecated, use "run"
instead

Basic Commands (Intermediate):
  get            Display one or many resources
  explain        Documentation of resources
  edit           Edit a resource on the server
  delete         Delete resources by filenames, stdin, resources and names, or by resources and
label selector

Deploy Commands:
  rollout        Manage the rollout of a resource
  rolling-update Perform a rolling update of the given ReplicationController
  scale          Set a new size for a Deployment, ReplicaSet, Replication Controller, or Job
  autoscale      Auto-scale a Deployment, ReplicaSet, or ReplicationController

Cluster Management Commands:
  certificate    Modify certificate resources.
  cluster-info   Display cluster info
  top            Display Resource (CPU/Memory/Storage) usage.
  cordon         Mark node as unschedulable
  uncordon       Mark node as schedulable
  drain          Drain node in preparation for maintenance
  taint          Update the taints on one or more nodes

Troubleshooting and Debugging Commands:
  describe       Show details of a specific resource or group of resources
  logs           Print the logs for a container in a pod
  attach         Attach to a running container
  exec           Execute a command in a container
  port-forward   Forward one or more local ports to a pod
  proxy          Run a proxy to the Kubernetes API server
  cp             Copy files and directories to and from containers.
  auth           Inspect authorization

Advanced Commands:
  apply          Apply a configuration to a resource by filename or stdin
  patch          Update field(s) of a resource using strategic merge patch
  replace        Replace a resource by filename or stdin
  convert        Convert config files between different API versions

Settings Commands:
  label          Update the labels on a resource
  annotate       Update the annotations on a resource
  completion     Output shell completion code for the specified shell (bash or zsh)

Other Commands:
  api-versions   Print the supported API versions on the server, in the form of "group/version"
  config         Modify kubeconfig files
  help           Help about any command
  plugin         Runs a command-line plugin
  version        Print the client and server version information

Use "kubectl <command> --help" for more information about a given command.
Use "kubectl options" for a list of global command-line options (applies to all commands).
$ kubectl version
Client Version: version.Info{Major:"1", Minor:"8", GitVersion:"v1.8.0", GitCommit:"6e937839ac04a38cac63e6a7a306c5d035fe7b0a", GitTreeState:"clean", BuildDate:"2017-09-28T22:57:57Z", GoVersion:"go1.8.3", Compiler:"gc", Platform:"linux/amd64"}
Server Version: version.Info{Major:"1", Minor:"5", GitVersion:"v1.5.2", GitCommit:"08e099554f3c31f6e6f07b448ab3ed78d0520507", GitTreeState:"clean", BuildDate:"1970-01-01T00:00:00Z", GoVersion:"go1.7.1", Compiler:"gc", Platform:"linux/amd64"}
$ kubectl get nodes
NAME      STATUS    ROLES     AGE       VERSION
host01    Ready     <none>    2m        v1.5.2
$ kubectl run kubernetes-bootcamp --image=docker.io/jocatalin/kubernetes-bootcamp:v1 --port=8080
deployment "kubernetes-bootcamp" created
$ kubectl get deployments
NAME                  DESIRED   CURRENT   UP-TO-DATE   AVAILABLE   AGE
kubernetes-bootcamp   1         1         1            1           36s
$ curl http://localhost:8001/version
{
  "major": "1",
  "minor": "5",
  "gitVersion": "v1.5.2",
  "gitCommit": "08e099554f3c31f6e6f07b448ab3ed78d0520507",
  "gitTreeState": "clean",
  "buildDate": "1970-01-01T00:00:00Z",
  "goVersion": "go1.7.1",
  "compiler": "gc",
  "platform": "linux/amd64"
}
$ export POD_NAME=$(kubectl get pods -o go-template --template '{{range .items}}{{.metadata.name}}{{"\n"}}{{end}}')
$ echo Name of the Pod: $POD_NAME
Name of the Pod: kubernetes-bootcamp-390780338-hcqwd
$ curl http://localhost:8001/api/v1/proxy/namespaces/default/pods/$POD_NAME/
Hello Kubernetes bootcamp! | Running on: kubernetes-bootcamp-390780338-hcqwd | v=1
```

### 3. Explore Your App

#### [Viewing Pods and Nodes](https://kubernetes.io/docs/tutorials/kubernetes-basics/explore-intro/) 

**Pods**，调度单位，代表了一组组应用容器资源，包括共享存储、网络、一些信息。每个节点上创建的都是Pod，kubernetes不直接操作容器。

![Pods](http://upload-images.jianshu.io/upload_images/5952841-e1fbef04290206da.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

[原图](https://d33wubrfki0l68.cloudfront.net/fe03f68d8ede9815184852ca2a4fd30325e5d15a/98064/docs/tutorials/kubernetes-basics/public/images/module_03_pods.svg)

**Nodes**，工作机器，一个Node可以包含多个Pods，受Master控制。其上运行了`kubelet`和具体的容器。

![Nodes](http://upload-images.jianshu.io/upload_images/5952841-e122156179a914a4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

[原图](https://d33wubrfki0l68.cloudfront.net/5cb72d407cbe2755e581b6de757e0d81760d5b86/a9df9/docs/tutorials/kubernetes-basics/public/images/module_03_nodes.svg)

#### [Interactive Tutorial - Exploring Your App](https://kubernetes.io/docs/tutorials/kubernetes-basics/explore-interactive/) 

最常用的kubectl命令如下：

- kubectl get - list resources
- kubectl describe - show detailed information about a resource
- kubectl logs - print the logs from a container in a pod
- kubectl exec - execute a command on a container in a pod

```shell
Kubernetes Bootcamp Terminal
$
$ sleep 1; ~/.bin/launch.sh
Starting Kubernetes...
Kubernetes Started
$ kubectl get pods
NAME                                  READY     STATUS    RESTARTS   AGE
kubernetes-bootcamp-390780338-dj3jf   1/1       Running   0          45s
$ kubectl describe pods
Name:           kubernetes-bootcamp-390780338-dj3jf
Namespace:      default
Node:           host01/172.17.0.102
Start Time:     Tue, 02 Jan 2018 06:48:57 +0000
Labels:         pod-template-hash=390780338
                run=kubernetes-bootcamp
Annotations:    kubernetes.io/created-by={"kind":"SerializedReference","apiVersion":"v1","reference":{"kind":"ReplicaSet","namespace":"default","name":"kubernetes-bootcamp-390780338","uid":"ff9981ed-ef88-11e7-bd3c-02...
Status:         Running
IP:             172.18.0.2
Created By:     ReplicaSet/kubernetes-bootcamp-390780338
Controlled By:  ReplicaSet/kubernetes-bootcamp-390780338
Containers:
  kubernetes-bootcamp:
    Container ID:   docker://38037e0c069c2f328d5b852efb0b9370edd1fb971e2500540a4c190b2f56e514
    Image:          docker.io/jocatalin/kubernetes-bootcamp:v1
    Image ID:       docker-pullable://jocatalin/kubernetes-bootcamp@sha256:0d6b8ee63bb57c5f5b6156f446b3bc3b3c143d233037f3a2f00e279c8fcc64af
    Port:           8080/TCP
    State:          Running
      Started:      Tue, 02 Jan 2018 06:48:59 +0000
    Ready:          True
    Restart Count:  0
    Environment:    <none>
    Mounts:
      /var/run/secrets/kubernetes.io/serviceaccount from default-token-d3wqp (ro)
Conditions:
  Type           Status
  Initialized    True
  Ready          True
  PodScheduled   True
Volumes:
  default-token-d3wqp:
    Type:        Secret (a volume populated by a Secret)
    SecretName:  default-token-d3wqp
    Optional:    false
QoS Class:       BestEffort
Node-Selectors:  <none>
Tolerations:     <none>
Events:
  Type    Reason     Age   From               Message
  ----    ------     ----  ----               -------
  Normal  Scheduled  1m    default-scheduler  Successfully assigned kubernetes-bootcamp-390780338-dj3jf to host01
  Normal  Pulled     1m    kubelet, host01    Container image "docker.io/jocatalin/kubernetes-bootcamp:v1" already present on machine
  Normal  Created    1m    kubelet, host01    Created container with docker id 38037e0c069c; Security:[seccomp=unconfined]
  Normal  Started    1m    kubelet, host01    Started container with docker id 38037e0c069c
}}{{end}}')D_NAME=$(kubectl get pods -o go-template --template '{{range .items}}{{.metadata.name}}{{"\n"
$ echo Name of the Pod: $POD_NAME
Name of the Pod: kubernetes-bootcamp-390780338-dj3jf
$ curl http://localhost:8001/api/v1/proxy/namespaces/default/pods/$POD_NAME/
Hello Kubernetes bootcamp! | Running on: kubernetes-bootcamp-390780338-dj3jf | v=1
$ kubectl logs $POD_NAME
Kubernetes Bootcamp App Started At: 2018-01-02T06:48:59.695Z | Running On:  kubernetes-bootcamp-390780338-dj3jf

Running On: kubernetes-bootcamp-390780338-dj3jf | Total Requests: 1 | App Uptime: 167.221 seconds | Log Time: 2018-01-02T06:51:46.916Z
$ kubectl exec $POD_NAME env
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
HOSTNAME=kubernetes-bootcamp-390780338-dj3jf
KUBERNETES_PORT_443_TCP_ADDR=10.0.0.1
KUBERNETES_SERVICE_HOST=10.0.0.1
KUBERNETES_SERVICE_PORT=443
KUBERNETES_SERVICE_PORT_HTTPS=443
KUBERNETES_PORT=tcp://10.0.0.1:443
KUBERNETES_PORT_443_TCP=tcp://10.0.0.1:443
KUBERNETES_PORT_443_TCP_PROTO=tcp
KUBERNETES_PORT_443_TCP_PORT=443
NPM_CONFIG_LOGLEVEL=info
NODE_VERSION=6.3.1
HOME=/root
$ kubectl exec -ti $POD_NAME bash
root@kubernetes-bootcamp-390780338-dj3jf:/# cat server.js
var http = require('http');
var requests=0;
var podname= process.env.HOSTNAME;
var startTime;
var host;
var handleRequest = function(request, response) {
  response.setHeader('Content-Type', 'text/plain');
  response.writeHead(200);
  response.write("Hello Kubernetes bootcamp! | Running on: ");
  response.write(host);
  response.end(" | v=1\n");
  console.log("Running On:" ,host, "| Total Requests:", ++requests,"| App Uptime:", (new Date() - startTime)/1000 , "seconds", "| Log Time:",new Date());
}
var www = http.createServer(handleRequest);
www.listen(8080,function () {
    startTime = new Date();;
    host = process.env.HOSTNAME;
    console.log ("Kubernetes Bootcamp App Started At:",startTime, "| Running On: " ,host, "\n" );
});
root@kubernetes-bootcamp-390780338-dj3jf:/# curl localhost:8080
Hello Kubernetes bootcamp! | Running on: kubernetes-bootcamp-390780338-dj3jf | v=1
root@kubernetes-bootcamp-390780338-dj3jf:/# exit
exit
```

### 4. Expose Your App Publicly

#### [Using a Service to Expose Your App](https://kubernetes.io/docs/tutorials/kubernetes-basics/expose-intro/) 

Pods拥有生命周期，一个Pod失败后，Master会自动重新启动一个。每个Pods拥有自己的IP地址，Services用来维护这些地址使得前端不会感知到后端IP地址的变化。

**Services**，定义了一个Pods的逻辑集合，以及对它们进行权限管理。

Services中Pod的端口不对集群外开放，Services的IP有以下几种方式：

- ClusterIP（默认） - 将服务公开在集群中的内部IP上。这种类型使服务只能从集群内访问。
- NodePort  - 使用nat在集群中每个选定节点的同一端口上公开该服务。使用`<nodeip>:<nodeport>`从集群外部访问服务。clusterip的超集。
- LoadBalancer  - 在当前云中创建一个外部负载平衡器（如果支持的话），并为该服务分配一个固定的外部IP。nodeport的超集。
- ExternalName  - 使用任意名称（通过规范中的`externalName`指定）通过返回具有名称的CNAME记录来公开该服务。没有代理被使用。这种类型需要v1.7或更高版本的kube-dns.

![Services](http://upload-images.jianshu.io/upload_images/5952841-b974fde91f047dcb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

[原图](https://d33wubrfki0l68.cloudfront.net/cc38b0f3c0fd94e66495e3a4198f2096cdecd3d5/ace10/docs/tutorials/kubernetes-basics/public/images/module_04_services.svg)

![Services and Labels](http://upload-images.jianshu.io/upload_images/5952841-f75ca15e6bab9dad.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

[原图](https://d33wubrfki0l68.cloudfront.net/b964c59cdc1979dd4e1904c25f43745564ef6bee/f3351/docs/tutorials/kubernetes-basics/public/images/module_04_labels.svg)

Services使用[labels and selectors](https://kubernetes.io/docs/concepts/overview/working-with-objects/labels)匹配一组Pods，这是一个允许在kubernetes中对对象进行逻辑操作的分组原语。标签是连接到对象的键/值对，可以以多种方式使用：

- 指定开发，测试和生产的对象
- 嵌入版本标签
- 使用标签对对象进行分类

#### [Interactive Tutorial - Exposing Your App](https://kubernetes.io/docs/tutorials/kubernetes-basics/expose-interactive/) 

```shell
Kubernetes Bootcamp Terminal
$
$ sleep 1; ~/.bin/launch.sh
Starting Kubernetes...
Kubernetes Started
$ kubectl get pods
NAME                                  READY     STATUS    RESTARTS   AGE
kubernetes-bootcamp-390780338-rvzm1   1/1       Running   0          12s
$ kubectl get services
NAME         TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)   AGE
kubernetes   ClusterIP   10.0.0.1     <none>        443/TCP   28s
$ kubectl expose deployment/kubernetes-bootcamp --type="NodePort" --port 8080
service "kubernetes-bootcamp" exposed
$ kubectl get services
NAME                  TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)          AGE
kubernetes            ClusterIP   10.0.0.1     <none>        443/TCP          1m
kubernetes-bootcamp   NodePort    10.0.0.240   <none>        8080:30165/TCP   6s
$ kubectl describe services/kubernetes-bootcamp
Name:                     kubernetes-bootcamp
Namespace:                default
Labels:                   run=kubernetes-bootcamp
Annotations:              <none>
Selector:                 run=kubernetes-bootcamp
Type:                     NodePort
IP:                       10.0.0.240
Port:                     <unset>  8080/TCP
TargetPort:               8080/TCP
NodePort:                 <unset>  30165/TCP
Endpoints:                172.18.0.2:8080
Session Affinity:         None
External Traffic Policy:  Cluster
Events:                   <none>
dePort}}')ODE_PORT=$(kubectl get services/kubernetes-bootcamp -o go-template='{{(index .spec.ports 0).no
$ echo NODE_PORT=$NODE_PORT
NODE_PORT=30165
$ curl host01:$NODE_PORT
Hello Kubernetes bootcamp! | Running on: kubernetes-bootcamp-390780338-rvzm1 | v=1
$ kubectl describe deployment
Name:                   kubernetes-bootcamp
Namespace:              default
CreationTimestamp:      Tue, 02 Jan 2018 08:33:18 +0000
Labels:                 run=kubernetes-bootcamp
Annotations:            deployment.kubernetes.io/revision=1
Selector:               run=kubernetes-bootcamp
Replicas:               1 desired | 1 updated | 1 total | 1 available | 0 unavailable
StrategyType:           RollingUpdate
MinReadySeconds:        0
RollingUpdateStrategy:  1 max unavailable, 1 max surge
Pod Template:
  Labels:  run=kubernetes-bootcamp
  Containers:
   kubernetes-bootcamp:
    Image:        docker.io/jocatalin/kubernetes-bootcamp:v1
    Port:         8080/TCP
    Environment:  <none>
    Mounts:       <none>
  Volumes:        <none>
Conditions:
  Type           Status  Reason
  ----           ------  ------
  Available      True    MinimumReplicasAvailable
OldReplicaSets:  <none>
NewReplicaSet:   <none>
Events:
  Type    Reason             Age   From                   Message
  ----    ------             ----  ----                   -------
  Normal  ScalingReplicaSet  3m    deployment-controller  Scaled up replica set kubernetes-bootcamp-390780338 to 1
$ kubectl get pods -l run=kubernetes-bootcamp
NAME                                  READY     STATUS    RESTARTS   AGE
kubernetes-bootcamp-390780338-rvzm1   1/1       Running   0          3m
$ kubectl get services -l run=kubernetes-bootcamp
NAME                  TYPE       CLUSTER-IP   EXTERNAL-IP   PORT(S)          AGE
kubernetes-bootcamp   NodePort   10.0.0.240   <none>        8080:30165/TCP   2m
}}{{end}}')D_NAME=$(kubectl get pods -o go-template --template '{{range .items}}{{.metadata.name}}{{"\n"
$ echo Name of the Pod: $POD_NAME
Name of the Pod: kubernetes-bootcamp-390780338-rvzm1
$ kubectl label pod $POD_NAME app=v1
pod "kubernetes-bootcamp-390780338-rvzm1" labeled
$ kubectl describe pods $POD_NAME
Name:           kubernetes-bootcamp-390780338-rvzm1
Namespace:      default
Node:           host01/172.17.0.54
Start Time:     Tue, 02 Jan 2018 08:33:22 +0000
Labels:         app=v1
                pod-template-hash=390780338
                run=kubernetes-bootcamp
Annotations:    kubernetes.io/created-by={"kind":"SerializedReference","apiVersion":"v1","reference":{"kind":"ReplicaSet","namespace":"default","name":"kubernetes-bootcamp-390780338","uid":"96d04963-ef97-11e7-a0ad-02...
Status:         Running
IP:             172.18.0.2
Created By:     ReplicaSet/kubernetes-bootcamp-390780338
Controlled By:  ReplicaSet/kubernetes-bootcamp-390780338
Containers:
  kubernetes-bootcamp:
    Container ID:   docker://48f886cdb8bb485f12b990b41503ad471874deeb1a8f557c8b46059035e61d17
    Image:          docker.io/jocatalin/kubernetes-bootcamp:v1
    Image ID:       docker-pullable://jocatalin/kubernetes-bootcamp@sha256:0d6b8ee63bb57c5f5b6156f446b3bc3b3c143d233037f3a2f00e279c8fcc64af
    Port:           8080/TCP
    State:          Running
      Started:      Tue, 02 Jan 2018 08:33:23 +0000
    Ready:          True
    Restart Count:  0
    Environment:    <none>
    Mounts:
      /var/run/secrets/kubernetes.io/serviceaccount from default-token-j5j6p (ro)
Conditions:
  Type           Status
  Initialized    True
  Ready          True
  PodScheduled   True
Volumes:
  default-token-j5j6p:
    Type:        Secret (a volume populated by a Secret)
    SecretName:  default-token-j5j6p
    Optional:    false
QoS Class:       BestEffort
Node-Selectors:  <none>
Tolerations:     <none>
Events:
  Type    Reason     Age   From               Message
  ----    ------     ----  ----               -------
  Normal  Scheduled  3m    default-scheduler  Successfully assigned kubernetes-bootcamp-390780338-rvzm1 to host01
  Normal  Pulled     3m    kubelet, host01    Container image "docker.io/jocatalin/kubernetes-bootcamp:v1" already present on machine
  Normal  Created    3m    kubelet, host01    Created container with docker id 48f886cdb8bb; Security:[seccomp=unconfined]
  Normal  Started    3m    kubelet, host01    Started container with docker id 48f886cdb8bb
$ kubectl get pods -l app=v1
NAME                                  READY     STATUS    RESTARTS   AGE
kubernetes-bootcamp-390780338-rvzm1   1/1       Running   0          4m
$ kubectl delete service -l run=kubernetes-bootcamp
service "kubernetes-bootcamp" deleted
$ kubectl get services
NAME         TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)   AGE
kubernetes   ClusterIP   10.0.0.1     <none>        443/TCP   4m
$ curl host01:$NODE_PORT
curl: (7) Failed to connect to host01 port 30165: Connection refused
$ kubectl exec -ti $POD_NAME curl localhost:8080
Hello Kubernetes bootcamp! | Running on: kubernetes-bootcamp-390780338-rvzm1 | v=1
```

### 5. Scale Your App

#### [Running Multiple Instances of Your App](https://kubernetes.io/docs/tutorials/kubernetes-basics/scale-intro/) 

**扩容**，更改部署的副本数量

![Scale1](http://upload-images.jianshu.io/upload_images/5952841-fabf81f9f51a8d1b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

[原文](https://d33wubrfki0l68.cloudfront.net/043eb67914e9474e30a303553d5a4c6c7301f378/0d8f6/docs/tutorials/kubernetes-basics/public/images/module_05_scaling1.svg)

![Scale2](http://upload-images.jianshu.io/upload_images/5952841-31255ecfc742d85f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

[原文](https://d33wubrfki0l68.cloudfront.net/30f75140a581110443397192d70a4cdb37df7bfc/b5f56/docs/tutorials/kubernetes-basics/public/images/module_05_scaling2.svg)

kubernetes支持自动规模缩放与增大，kubernetes自带负载均衡。

#### [Interactive Tutorial - Scaling Your App](https://kubernetes.io/docs/tutorials/kubernetes-basics/scale-interactive/) 

```shell
Kubernetes Bootcamp Terminal
$
$ sleep 1; ~/.bin/launch.sh
Starting Kubernetes...
Kubernetes Started
$ kubectl get deployments
NAME                  DESIRED   CURRENT   UP-TO-DATE   AVAILABLE   AGE
kubernetes-bootcamp   1         1         1            0           11s
$ kubectl scale deployments/kubernetes-bootcamp --replicas=4
deployment "kubernetes-bootcamp" scaled
$ kubectl get deployments
NAME                  DESIRED   CURRENT   UP-TO-DATE   AVAILABLE   AGE
kubernetes-bootcamp   4         4         4            1           1m
$ kubectl get pods -o wide
NAME                                  READY     STATUS              RESTARTS   AGE       IP           NODE
kubernetes-bootcamp-390780338-18wjr   1/1       Running             0          1m        172.18.0.2   host01
kubernetes-bootcamp-390780338-g7dkf   0/1       ContainerCreating   0          11s       <none>       host01
kubernetes-bootcamp-390780338-jzpst   0/1       ContainerCreating   0          11s       <none>       host01
kubernetes-bootcamp-390780338-tm3vw   0/1       ContainerCreating   0          11s       <none>       host01
$ kubectl describe deployments/kubernetes-bootcamp
Name:                   kubernetes-bootcamp
Namespace:              default
CreationTimestamp:      Tue, 02 Jan 2018 09:09:58 +0000
Labels:                 run=kubernetes-bootcamp
Annotations:            deployment.kubernetes.io/revision=1
Selector:               run=kubernetes-bootcamp
Replicas:               4 desired | 4 updated | 4 total | 4 available | 0 unavailable
StrategyType:           RollingUpdate
MinReadySeconds:        0
RollingUpdateStrategy:  1 max unavailable, 1 max surge
Pod Template:
  Labels:  run=kubernetes-bootcamp
  Containers:
   kubernetes-bootcamp:
    Image:        docker.io/jocatalin/kubernetes-bootcamp:v1
    Port:         8080/TCP
    Environment:  <none>
    Mounts:       <none>
  Volumes:        <none>
Conditions:
  Type           Status  Reason
  ----           ------  ------
  Available      True    MinimumReplicasAvailable
OldReplicaSets:  <none>
NewReplicaSet:   <none>
Events:
  Type    Reason             Age   From                   Message
  ----    ------             ----  ----                   -------
  Normal  ScalingReplicaSet  1m    deployment-controller  Scaled up replica set kubernetes-bootcamp-390780338 to 1
  Normal  ScalingReplicaSet  25s   deployment-controller  Scaled up replica set kubernetes-bootcamp-390780338 to 4
$ kubectl describe services/kubernetes-bootcamp
Name:                     kubernetes-bootcamp
Namespace:                default
Labels:                   run=kubernetes-bootcamp
Annotations:              <none>
Selector:                 run=kubernetes-bootcamp
Type:                     NodePort
IP:                       10.0.0.119
Port:                     <unset>  8080/TCP
TargetPort:               8080/TCP
NodePort:                 <unset>  30169/TCP
Endpoints:                172.18.0.2:8080,172.18.0.3:8080,172.18.0.4:8080 + 1 more...
Session Affinity:         None
External Traffic Policy:  Cluster
Events:                   <none>
dePort}}')ODE_PORT=$(kubectl get services/kubernetes-bootcamp -o go-template='{{(index .spec.ports 0).no
$ echo NODE_PORT=$NODE_PORT
NODE_PORT=30169
$ curl host01:$NODE_PORT
Hello Kubernetes bootcamp! | Running on: kubernetes-bootcamp-390780338-tm3vw | v=1
$ kubectl scale deployments/kubernetes-bootcamp --replicas=2
deployment "kubernetes-bootcamp" scaled
$ kubectl get deployments
NAME                  DESIRED   CURRENT   UP-TO-DATE   AVAILABLE   AGE
kubernetes-bootcamp   2         2         2            2           1m
$ kubectl get pods -o wide
NAME                                  READY     STATUS        RESTARTS   AGE       IP           NODE
kubernetes-bootcamp-390780338-18wjr   1/1       Running       0          1m        172.18.0.2   host01
kubernetes-bootcamp-390780338-g7dkf   1/1       Terminating   0          50s       172.18.0.5   host01
kubernetes-bootcamp-390780338-jzpst   1/1       Running       0          50s       172.18.0.3   host01
kubernetes-bootcamp-390780338-tm3vw   1/1       Terminating   0          50s       172.18.0.4   host01
```

### 6. Update Your App

#### [Performing a Rolling Update](https://kubernetes.io/docs/tutorials/kubernetes-basics/update-intro/)

滚动升级，不需要停机。

版本升级可以回退到任意一个版本。

升级示意图如下

![滚动升级1](http://upload-images.jianshu.io/upload_images/5952841-bbd975cf79ce5874.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

[原图](https://d33wubrfki0l68.cloudfront.net/30f75140a581110443397192d70a4cdb37df7bfc/fa906/docs/tutorials/kubernetes-basics/public/images/module_06_rollingupdates1.svg)

![滚动升级2](http://upload-images.jianshu.io/upload_images/5952841-a67b7fc950f02ba3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

[原图](https://d33wubrfki0l68.cloudfront.net/678bcc3281bfcc588e87c73ffdc73c7a8380aca9/703a2/docs/tutorials/kubernetes-basics/public/images/module_06_rollingupdates2.svg)

![滚动升级3](http://upload-images.jianshu.io/upload_images/5952841-cdbc2fd22b4869aa.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

[原图](https://d33wubrfki0l68.cloudfront.net/9b57c000ea41aca21842da9e1d596cf22f1b9561/91786/docs/tutorials/kubernetes-basics/public/images/module_06_rollingupdates3.svg)

![滚动升级4](http://upload-images.jianshu.io/upload_images/5952841-7991680fdf7acb64.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

[原图](https://d33wubrfki0l68.cloudfront.net/6d8bc1ebb4dc67051242bc828d3ae849dbeedb93/fbfa8/docs/tutorials/kubernetes-basics/public/images/module_06_rollingupdates4.svg)

#### [Interactive Tutorial - Updating Your App](https://kubernetes.io/docs/tutorials/kubernetes-basics/update-interactive/)

[日志](https://gitee.com/liqiang311/codes/je5gsr960c37qlviaud2415)

# [Setup](https://kubernetes.io/docs/setup/)

##  [Picking the Right Solution](https://kubernetes.io/docs/setup/pick-right-solution/)

### Local-machine Solutions

- [Minikube](https://kubernetes.io/docs/getting-started-guides/minikube/)适用于创建本地的、单节点集群。
- [Kubeadm-dind](https://github.com/Mirantis/kubeadm-dind-cluster) is a multi-node (while minikube is single-node) Kubernetes cluster which only requires a docker daemon. It uses docker-in-docker technique to spawn the Kubernetes cluster.
- [Ubuntu on LXD](https://kubernetes.io/docs/getting-started-guides/ubuntu/local/) supports a nine-instance deployment on localhost.
- [IBM Cloud Private-CE (Community Edition)](https://github.com/IBM/deploy-ibm-cloud-private) can use VirtualBox on your machine to deploy Kubernetes to one or more VMs for development and test scenarios. Scales to full multi-node cluster.

### Hosted Solutions

## Independent Solutions

### [Running Kubernetes Locally via Minikube](https://kubernetes.io/docs/getting-started-guides/minikube/) 



### Bootstrapping Clusters with kubeadm

#### [Installing kubeadm](https://kubernetes.io/docs/setup/independent/install-kubeadm/) 

#### [Using kubeadm to Create a Cluster](https://kubernetes.io/docs/setup/independent/create-cluster-kubeadm/) 

#### [Troubleshooting kubeadm](https://kubernetes.io/docs/setup/independent/troubleshooting-kubeadm/) 

### [Creating a Custom Cluster from Scratch](https://kubernetes.io/docs/getting-started-guides/scratch/) 

### [Deprecated Alternatives](https://kubernetes.io/docs/getting-started-guides/alternatives/)
