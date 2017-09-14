---
title: mysql入门
date: 2017-09-14 15:59:00
tags: mysql
categories: sql
---

## 功能性命令

<!-- more -->

连接MySQL`mysql -u root -p`

创建数据库`create database <db name>;`

连接到数据库`use <db name>`

创建表
```
create table <table name> (
<字段名> <字段属性>
);
```

插入数据`insert into <表名> [( <字段名1>[,..<字段名n > ])] values ( 值1 )[, ( 值n )]

## 查看命令

查看MySQL版本`select version();`

查看所有数据库`show databases;`

查看当前数据库`select database();`