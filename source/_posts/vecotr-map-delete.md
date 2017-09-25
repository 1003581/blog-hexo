---
title: C++ STL vector和map删除方法
date: 2017-09-14 16:00:18
tags: stl
categories: c++
---

# vector

<!-- more -->

## 删除

```c++
vector<int> v(10,0);
for (vector<int>::iterator it=v.begin(); it!=v.end(); )
{
    if (删除条件)
    {
        it = v.erase(it);
    }
else
    {
        it ++;
    }
}
```

# map

## 删除

```c++
map<int,int> m;
for (map<int,int>::iterator it=m.begin(); it!=m.end(); )
{
    if (删除条件)
    {
        m.erase(it++);
    }
    else
    {
        it ++;
    }
}
```
