---
title: 定时提交代码到gitlab
date: 2017-09-14 15:11:16
tags: tool
categories: tool
---

## 定时提交代码

<!-- more -->

**用于同步笔记到GitLab, 不适合用于生产环境代码**.

### Windows

1. 打开`任务计划程序`.
1. 右方点击`创建基本任务`.
1. 输入`任务名称`, `运行时间`.
1. 选择执行`启动程序`, 在某个位置新建脚本文件`schedule.bat`, 输入如下
    ```
    e:
    cd E:\GitLab\note
    git status
    git add .
    git commit -m "auto update"
    git push origin master
    ```
1. 其它默认, 点击完成, 左边点击 `任务计划程序库` , 右键刚刚创建的任务, 点击`运行`.

注: 若用https连接, 则要走在.config中设置密码 

```
[remote "origin"]
url = git@github.com:USERNAME/REPONAME.git  
```

### Linux

采用Crontab定时任务

新建执行脚本文件`schedule.sh`

```
cd gitlab/note
git status
git add .
git commit -m "auto update"
git push origin master
```

每天3点钟运行

```
0 3 * * * bash schedule.sh
```