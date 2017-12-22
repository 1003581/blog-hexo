---
title: mysql入门
date: 2017-09-14 15:59:00
tags: mysql
categories: sql
---

## 功能性命令

<!-- more -->

连接MySQL`mysql -u root -p` `mysql -uroot -p123456`

创建数据库`create database <db name>;`

连接到数据库`use <db name>`

创建表
```
create table <table name> (
<字段名> <字段属性>
);
```

主机上导出数据库
```
docker exec gpuswork_mysql_1  mysqldump -uroot -p123456 gpu > /tmp/backup.sql
# gpu后可选择添加表名，可以到处指定表
```

主机上导入数据库
```
docker cp /tmp/backup.sql gputest_mysql_1:/tmp/
docker exec gputest_mysql_1 mysql -uroot -p123456 gpu -e "source /tmp/backup.sql"
```

## 查看命令

查看MySQL版本`select version();`

查看所有数据库`show databases;`

查看当前数据库`select database();`

## 增

插入数据
```
insert into <表名> [( <字段名1>[,..<字段名n > ])] values ( 值1 )[, ( 值n )];
```

## 删

```
delete from <table> where condition;
```

## 改

## 查