---
title: openresty遇到的问题
date: 2017-09-14 16:02:05
tags: openresty
categories: openresty
---

## 问题

<!-- more -->

### WebSocket代理

[解决方法](http://www.cnblogs.com/mfrbuaa/p/5413786.html)

在入口nginx.conf中添加配置

```
    map $http_upgrade $connection_upgrade {
        default upgrade;
        '' close;
    }
```

```
        # add for websocket of paas
        location ^~ /keybox/ {
            proxy_pass http://10.96.32.66;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "Upgrade";
        }
```

### 双层Nginx URI被修改问题

初始请求URI

```
/winery/servicetemplates/http%253A%252F%252Fwww.zte.com.cn%252Fpaas%252Fr3%252Fservice%252Fopenabtest%252Flatest/112/topologytemplate/?edit
```

到第二层后变为解码后的([解码地址](http://tool.chinaz.com/tools/urlencode.aspx))

```
/winery/servicetemplates/http%3A%2F%2Fwww.zte.com.cn%2Fpaas%2Fr3%2Fservice%2Fopenabtest%2Flatest/112/topologytemplate/?edit
```



测试如下：

本地一层nginx代理成功，无此问题

将Access层去掉后，无此问题

问题定位Access层。

中间日志设置无效。

在第二层的location接口修改

```nginx
location /winery/ {
    return 200 $request_uri;
    #proxy_pass http://10.96.32.66/winery/;
}

```

这样就会收到的request请求是否正确

然后利用curl命令访问第一层

```
curl localhost:8384/winery/servicetemplates/http%253A%252F%252Fwww.zte.com.cn%252Fpaas%252Fr3%252Fservice%252Fopenabtest%252Flatest/112/topologytemplate/?edit
```

发现返回如下

```
/winery/servicetemplates/http%3A%2F%2Fwww.zte.com.cn%2Fpaas%2Fr3%2Fservice%2Fopenabtest%2Flatest/112/topologytemplate/?edit
```

解决办法：

```
        location ~* (.*) {
            set $destIP '';
            set $trace_log_seq '';
            set $trace_log_depth '';
            rewrite_by_lua_file conf/lua/gray_user_filter.lua;
            proxy_set_header trace-log-seq $trace_log_seq;
            proxy_set_header trace-log-depth $trace_log_depth;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header REMOTE-HOST $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_connect_timeout 600s;
            proxy_read_timeout 600s;
            proxy_send_timeout 600s;
            proxy_pass http://$destIP:8384/$1?$args;
            proxy_redirect   http://$destIP:8384/ /;
        }
```

`$1`是正则匹配中第一个括号内的东西，`args`是?后的参数。

发现正则匹配中会将uri进行解码，导致`$1`中放入解码后的文本。

解决方法1：

将15行的`/$1?$args`改为`$request_uri`

解决方法2：

将1行的location改为`location / {`，同时将15行的`/$1?$args`删除。



修改后curl返回为

```
/winery/servicetemplates/http%253A%252F%252Fwww.zte.com.cn%252Fpaas%252Fr3%252Fservice%252Fopenabtest%252Flatest/112/topologytemplate/?edit
```

