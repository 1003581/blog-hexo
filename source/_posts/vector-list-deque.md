---
title: C++ STL vector、list、deque的比较
date: 2017-09-14 16:00:07
tags: STL
categories: c++
---

# STL容器

<!-- more -->

顺序存储结构：vector、list、deque

关联存储结构：set、multiset、map、multimap

deque: double end-queue 双端队列

# vector

内存方面：

内存地址连续，相当于一个数组

访问效率方面：

高效随机访问（通过[]访问）

插入删除效率方面：

在尾端效率很高、其他位置效率低下

# list

内存方面：

双向链表，非连续内存

访问效率方面：

随机访问效率低下

插入删除效率方面：

随机插入删除效率高

# deque

内存方面：

整合vector和list，两级数据结构，第一级为list，list里保存多个vector

访问效率方面：

方面，支持[]访问

插入删除效率方面：

两端插入删除效率高

# 总结

1. 若需要随机访问操作，则选择vector；
2. 若已经知道需要存储元素的数目，则选择vector；
3. 若需要随机插入/删除（不仅仅在两端），则选择list
4. 只有需要在首端进行插入/删除操作的时候，还要兼顾随机访问效率，才选择deque，否则都选择vector。
5. 若既需要随机插入/删除，又需要随机访问，则需要在vector与list间做个折中-deque。
6. 当要存储的是大型负责类对象时，list要优于vector；当然这时候也可以用vector来存储指向对象的指针，同样会取得较高的效率，但是指针的维护非常容易出错，因此不推荐使用。
