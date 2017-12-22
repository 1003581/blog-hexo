---
title: openresty遇到的问题
date: 2017-09-14 16:02:05
tags: openresty
categories: openresty
---

## 资料

- [OpenResty® - 中文官方站](OpenResty® - 中文官方站)
- [OpenResty Github](https://github.com/openresty/openresty)
- [openresty/lua-nginx-module GitHub](https://github.com/openresty/lua-nginx-module)
- [OpenResty 最佳实践-极客学院Wiki](http://wiki.jikexueyuan.com/project/openresty/)
- [LuaJIT函数优化列表](http://wiki.luajit.org/NYI)
- [Lua 在线工具 | 菜鸟工具](https://c.runoob.com/compile/66)
- [基于nginx的api网关 - akin的博客 - 博客频道 - CSDN.NET](http://blog.csdn.net/akin_zhou/article/details/50373414)
- [服务端架构中的“网关服务器” - 永远的朋友 - 51CTO技术博客](http://yaocoder.blog.51cto.com/2668309/1374280/)
- [Nginx配置文件（nginx.conf）配置详解 - 浮云中的神马 - 博客频道 - CSDN.NET](http://blog.csdn.net/tjcyjd/article/details/50695922)
- [第二章 OpenResty(Nginx+Lua)开发入门 - 开涛的博客—公众号：kaitao-1234567，一如既往的干货分享 - ITeye技术网站](http://jinnianshilongnian.iteye.com/blog/2186448)
- [Nginx担当WebSockets代理 - mfrbuaa - 博客园](http://www.cnblogs.com/mfrbuaa/p/5413786.html)
- [UrlEncode编码/UrlDecode解码 - 站长工具](http://tool.chinaz.com/tools/urlencode.aspx)
- [如何在nginx中使用系统的环境变量(转) - 菜鸟刚起飞 - 博客频道 - CSDN.NET](http://blog.csdn.net/joker_zhou/article/details/49361753)
- [请教一下关于 Nginx 反向代理到虚拟目录时如何处理使用绝对路径的静态文件 - V2EX](https://www.v2ex.com/t/129032)
- [nginx常用代理配置 - Florian - 博客园](http://www.cnblogs.com/fanzhidongyzby/p/5194895.html)
- [Module ngx_http_sub_module](http://nginx.org/en/docs/http/ngx_http_sub_module.html)
- [编译nginx的源码安装subs_filter模块 - dudu - 博客园](编译nginx的源码安装subs_filter模块 - dudu - 博客园)

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

