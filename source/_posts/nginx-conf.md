---
title: nginx配置文件详解
date: 2017-09-14 16:01:54
tags: nginx
categories: openresty
---

##  nginx.conf

<!-- more -->

[参照](http://blog.csdn.net/tjcyjd/article/details/50695922)

```nginx
# 定义Nginx运行的用户和用户组。window下不指定
user www www;

# nginx进程数，建议设置为等于CPU总核心数
worker_processes 1;

# 全局错误日志定义类型，[ debug | info | notice | warn | error | crit ]
error_log /var/log/ngingx/error.log warn;

# 进程文件
pid /var/run/nginx.pid;

# 一个nginx进程打开的最多文件描述符数目，理论值应该是最多打开文件数（系统的值ulimit -n）
# 与nginx进程数相除，但是nginx分配请求并不均匀，所以建议与ulimit -n的值保持一致
worker_rlimit_nofile 1024;

#工作模式与连接数上限
events{
    # 参考事件模型，use [ kqueue | rtsig | epoll | /dev/poll | select | poll ];
    # epoll模型是Linux 2.6以上版本内核中的高性能网络I/O模型
    # 如果跑在FreeBSD上面，就用kqueue模型，Windows下不指定
    use epoll;

    # 每个工作进程的最大连接数量。根据硬件调整，和前面工作进程配合起来用，尽量大，
    # 但是别把cpu跑到100%就行。每个进程允许的最多连接数，理论上每台nginx服务器的最大连接数
    # 为worker_processes*worker_connections
    worker_connections 204800;

    # keepalive超时时间
    keepalive_timeout 60;

    # 为打开文件的指定缓存，默认是没有启用的，max指定缓存数量，建议和打开文件数一致，
    # inactive是指经过多长时间文件没被请求后删除缓存。
    open_file_cache max=1024 inactive=60s;

    # 多长时间检查一次缓存的有效信息。
    open_file_cache_valid 80s;

    # open_file_cache指令中的inactive参数时间内文件的最少使用次数，如果超过这个数字，文件
    # 描述符一直是在缓存中打开的，如上例，如果有一个文件在inactive时间内一次没被使用，它将被移除。
    open_file_cache_min_uses 1;
}

# 设定http服务器，利用它的反向代理功能提供负载均衡支持
http{
    # 设定mime类型,类型由mime.type文件定义，该文件在为nginx自带，且与nginx.conf在同一目录下。
    include mime.types;

    # 默认文件类型
    default_type application/octet-stream;

    # 默认编码
    charset utf-8;

    # 保存服务器名字的hash表是由指令server_names_hash_max_size 和
    # server_names_hash_bucket_size所控制的。如果Nginx给出需要增大hash max size 或
    # hash bucket size的提示，那么首要的是增大前一个参数的大小.
    server_names_hash_bucker_size 128;

    # 客户端请求头部的缓冲区大小。这个可以根据你的系统分页大小来设置，一般一个请求的头部大小
    # 不会超过1k，不过由于一般系统分页都要大于1k，所以这里设置为分页大小。
    # 分页大小可以用命令getconf PAGESIZE取得。
    client_header_buffer_size 4k;

    # 客户请求头缓冲大小。nginx默认会用client_header_buffer_size这个buffer来读取
    # header值，如果header过大，它会使用large_client_header_buffers来读取。
    large_client_header_buffers 8 128k;

    # 设定通过nginx上传文件的大小
    client_max_body_size 300m;

    # 这个指令指定缓存是否启用。
    # 例: open_file_cache max=1000 inactive=20s;
    # open_file_cache_valid 30s;
    # open_file_cache_min_uses 2;
    # open_file_cache_errors on;
    # 语法:open_file_cache_errors on | off
    # 默认值:open_file_cache_errors off
    # 使用字段:http, server, location 这个指令指定是否在搜索一个文件是记录cache错误.
    open_file_cache max=102400 inactive=20s;

    # 日志格式设置。
    # $remote_addr与$http_x_forwarded_for用以记录客户端的ip地址；
    # $remote_user：用来记录客户端用户名称；
    # $time_local： 用来记录访问时间与时区；
    # $request： 用来记录请求的url与http协议；
    # $status： 用来记录请求状态；成功是200，
    # $body_bytes_sent ：记录发送给客户端文件主体内容大小；
    # $http_referer：用来记录从那个页面链接访问过来的；
    # $http_user_agent：记录客户浏览器的相关信息；
    # 通常web服务器放在反向代理的后面，这样就不能获取到客户的IP地址了，通过$remote_add
    # 拿到的IP地址是反向代理服务器的iP地址。反向代理服务器在转发请求的http头信息中，
    # 可以增加x_forwarded_for信息，用以记录原有客户端的IP地址和原来客户端的请求的服务器地址。
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    "$http_user_agent" "$http_x_forwarded_for"';
    log_format log404   '$status [$time_local] $remote_addr $host$request_uri'
                        '$sent_http_location';

    # 用了log_format指令设置了日志格式之后，需要用access_log指令指定日志文件的存放路径
    access_log logs/access.log main;
    access_log logs/access.404.log  log404;

    # sendfile指令指定 nginx 是否调用sendfile 函数（zero copy 方式）来输出文件，
    # 对于普通应用，必须设为on。如果用来进行下载等应用磁盘IO重负载应用，可设置为off，
    # 以平衡磁盘与网络IO处理速度，降低系统uptime。
    sendfile on;

    # 此选项允许或禁止使用socke的TCP_CORK的选项，此选项仅在使用sendfile的时候使用
    tcp_nopush on;

    # keepalive超时时间
    keepalive_timeout 65;

    # 开启gzip压缩输出
    gzip on;  

    # 配置文件的包含
    include /etc/nginx/conf.d/*.conf

    # 配置DNS解析服务器，可以多个，用空格分隔
    resolver 10.30.1.9;
}
# 生产环境中不要使用"daemon"和"master_process"指令，这些选项仅用于开发调试。
daemon off;
```

## 日志

日志级别如下：

```
ngx.STDERR     -- 标准输出
ngx.EMERG      -- 紧急报错
ngx.ALERT      -- 报警
ngx.CRIT       -- 严重，系统故障，触发运维告警系统
ngx.ERR        -- 错误，业务不可恢复性错误
ngx.WARN       -- 告警，业务中可忽略错误
ngx.NOTICE     -- 提醒，业务比较重要信息
ngx.INFO       -- 信息，业务琐碎日志信息，包含不同情况判断等
ngx.DEBUG      -- 调试
```

修改nginx.conf文件中

```
error_log /var/log/ngingx/error.log warn
```

然后在Lua脚本中调用

```
ngx.log(ngx.WARN,string.format(""))
ngx.log(ngx.ERR,string.format(""))
```

日志输出如下：

```
2017/06/06 19:32:26 [warn] 19356#3272: *20 [lua] content_by_lua(nginx.conf:52):5: [WARN] The request is from 127.0.0.1, client: 127.0.0.1, server: , request: "GET / HTTP/1.1", host: "localhost"
2017/06/06 19:32:26 [error] 19356#3272: *20 [lua] content_by_lua(nginx.conf:52):6: [ERROR] The request is from 127.0.0.1, client: 127.0.0.1, server: , request: "GET / HTTP/1.1", host: "localhost"
127.0.0.1 - - [06/Jun/2017:19:32:26 +0800] "GET / HTTP/1.1" 200 41 "-" "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.1; Win64; x64; Trident/4.0; .NET CLR 2.0.50727; SLCC2; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E; TCO_20170606192303)" "-"
```

第一行为warn级别，第二行为error级别，第三行为Access日志。

系统中可以将关键流程日志设置为Warn，而非Notice和Info，因为nginx本身会输出大量的Notice和info信息。

## location 匹配规则

语法规则

```
location [=|~|~*|^~] /uri/ { … }
```

模式 | 含义
--- | ---
`location = /uri` | = 表示精确匹配，只有完全匹配上才能生效
`location ^~ /uri`  |  ^~ 开头对URL路径进行前缀匹配，并且在正则之前。
`location ~ pattern` | 开头表示区分大小写的正则匹配
`location ~* pattern` | 开头表示不区分大小写的正则匹配
`location /uri`  | 不带任何修饰符，也表示前缀匹配，但是在正则匹配之后
`location /` | 通用匹配，任何未匹配到其它location的请求都会匹配到，相当于switch中的default

前缀匹配时，Nginx 不对 url 做编码，因此请求为 /static/20%/aa，可以被规则 ^~ /static/ /aa 匹配到（注意是空格）
多个 location 配置的情况下匹配顺序为

- 首先精确匹配 =
- 其次前缀匹配 ^~
- 其次是按文件中顺序的正则匹配
- 然后匹配不带任何修饰的前缀匹配。
- 最后是交给 / 通用匹配
- 当有匹配成功时候，停止匹配，按当前匹配规则处理请求

注意：前缀匹配，如果有包含关系时，按最大匹配原则进行匹配。比如在前缀匹配：`location /dir01` 与 `location /dir01/dir02`，如有请求 `http://localhost/dir01/dir02/file` 将最终匹配到 `location /dir01/dir02`

