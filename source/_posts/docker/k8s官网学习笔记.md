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

# [Tutorials](https://kubernetes.io/docs/tutorials/)

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



#### [Interactive Tutorial - Deploying an App](https://kubernetes.io/docs/tutorials/kubernetes-basics/deploy-interactive/) 

### 3. Explore Your App

#### [Viewing Pods and Nodes](https://kubernetes.io/docs/tutorials/kubernetes-basics/explore-intro/) 
#### [Interactive Tutorial - Exploring Your App](https://kubernetes.io/docs/tutorials/kubernetes-basics/explore-interactive/) 

### 4. Expose Your App Publicly

#### [Using a Service to Expose Your App](https://kubernetes.io/docs/tutorials/kubernetes-basics/expose-intro/) 
#### [Interactive Tutorial - Exposing Your App](https://kubernetes.io/docs/tutorials/kubernetes-basics/expose-interactive/) 

### 5. Scale Your App

#### [Running Multiple Instances of Your App](https://kubernetes.io/docs/tutorials/kubernetes-basics/scale-intro/) 
#### [Interactive Tutorial - Scaling Your App](https://kubernetes.io/docs/tutorials/kubernetes-basics/scale-interactive/) 

### 6. Update Your App
