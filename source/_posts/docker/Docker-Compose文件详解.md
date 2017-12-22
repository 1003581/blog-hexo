---
title: Docker Compose文件详解
date: 2017-09-14 15:12:00
tags: docker
categories: docker
---

[官方文档](https://docs.docker.com/compose/overview/)

<!-- more -->

## Compose File

[compose文件版本与Docker版本的对应关系](https://docs.docker.com/compose/compose-file/compose-versioning/#compatibility-matrix)

### Version 3

#### build

```
build: .
image: webapp:tag
```

在当前目录构建镜像，镜像名为`webapp:tag`

```
build:
  context: ./build
  dockerfile: dockerfile-alternate
  args:
    buildno: 1
```

##### context

指向了一个包含Dockerfile的目录（可以是相对目录），或者指向一个Git仓库的url。

##### dockerfile

指定了Alternate Dockerfile

##### args

首先在Dockerfile中列出参数，如

```
ARG buildno
ARG password

RUN echo "Build number: $buildno"
RUN script-requiring-password.sh "$password"
```

然后在compose文件中对参数进行赋值，如下。同时支持map或者列表格式。

```
build:
  context: .
  args:
    buildno: 1
    password: secret

build:
  context: .
  args:
    - buildno=1
    - password=secret
```

##### cache_from

new in v3.2

```
build:
  context: .
  cache_from:
    - alpine:latest
    - corp/web_app:3.14
```

##### labels

new in v3.3，参照[Docker Labels](https://docs.docker.com/engine/userguide/labels-custom-metadata/#value-guidelines)

```
build:
  context: .
  labels:
    com.example.description: "Accounting webapp"
    com.example.department: "Finance"
    com.example.label-with-empty-value: ""


build:
  context: .
  labels:
    - "com.example.description=Accounting webapp"
    - "com.example.department=Finance"
    - "com.example.label-with-empty-value"
```

#### cap_add,cap_drop

添加和删除权限，所有权限命令`man 7 capabilities`，解释[en](https://linux.die.net/man/7/capabilities)

```
cap_add:
  - ALL

cap_drop:
  - NET_ADMIN
  - SYS_ADMIN
```

在[`deploying a stack in swarm mode`](https://docs.docker.com/engine/reference/commandline/stack_deploy/)模式下失效

#### command

对默认命令进行覆盖重写

```
command: bundle exec thin -p 3000
command: ["bundle", "exec", "thin", "-p", "3000"]
```

#### configs

new in v3.3，在各个服务下面配置configs，然后最后在与`service`平级的`configs`下指定具体值。

##### short syntax

```
version: "3.3"
services:
  redis:
    image: redis:latest
    deploy:
      replicas: 1
    configs:
      - my_config
      - my_other_config
configs:
  my_config:
    file: ./my_config.txt
  my_other_config:
    external: true
```

仅仅指定`config name`，`redis`服务可以访问`my_config`和`my_other_config`，`my_config`具体的值保存在`./myconfig.txt`中。这些将被挂载到容器内部的`/my_config`和`/my_other_config`中。

##### long syntax

```
version: "3.3"
services:
  redis:
    image: redis:latest
    deploy:
      replicas: 1
    configs:
      - source: my_config
        target: /redis_config
        uid: '103'
        gid: '103'
        mode: 0440
configs:
  my_config:
    file: ./my_config.txt
  my_other_config:
    external: true
```

- `source`:Docker中存在的配置名称
- `target`:将要在容器中挂载的目录，默认为`/<source>`
- `uid`和`gid`:配置文件在容器中的所属用户和所属组别，Linux默认为0(root)，Windows不支持。
- `mode`:挂载文件的权限，默认为`0444`，因为不可写，所以设置为写权限为无用的，可以设置为执行x权限

#### cgroup_parent

指定容器的cgroup parent，在`deploying a stack in swarm mode with`下无效。[Cgroup介绍](http://www.cnblogs.com/lisperl/archive/2012/04/17/2453838.html)

```
cgroup_parent: m-executor-abcd
```

#### container_name

指定容器name，用来代替默认名称

```
container_name: my-web-container
```

#### credential_spec

new in v3.3，仅在Windows下使用

```
credential_spec:
  file: c:/WINDOWS/my-credential-spec.txt

credential_spec:
  registry: HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Virtualization\Containers\CredentialSpecs
```

#### deploy

v3新增功能，用来部署成`swarm`模式。被`docker-compose up`和`docker-compose run`指令忽略。

```
version: '3'
services:
  redis:
    image: redis:alpine
    deploy:
      replicas: 6
      update_config:
        parallelism: 2
        delay: 10s
      restart_policy:
        condition: on-failure
```

##### mode

`global`或者`replicated`，默认为`replicated`,区别[replicated](https://docs.docker.com/engine/swarm/how-swarm-mode-works/services/#replicated-and-global-services)

##### replicas

如果mode为replicated（默认），则该字段指定在任意时间上应该被运行的容器数目。

##### placement

同docker启动的constaints参数

```
version: '3'
services:
  db:
    image: postgres
    deploy:
      placement:
        constraints:
          - node.role == manager
          - engine.labels.operatingsystem == ubuntu 14.04
```

##### update_config

指定服务如何被更新:

- `parallelism`指定一次升级的容器数目
- `delay`指定每组容器升级间的间隙时间
- `failure_action`指定更新失败时的动作,`continue`or`pause`(default)
- `monitor`指定为了监视每个更新任务的失败而持续的监视时间(`ns|us|ms|s|m|h`),默认0s
- `max_failure_ratio`指定升级期间所容忍的失败率

```
version: '3'
services:
  vote:
    image: dockersamples/examplevotingapp_vote:before
    depends_on:
      - redis
    deploy:
      replicas: 2
      update_config:
        parallelism: 2
        delay: 10s
```

##### resources

对资源进行管控，v3版本之前是`(cpu_shares, cpu_quota, cpuset, mem_limit, memswap_limit, mem_swappiness)`

```
version: '3'
services:
  redis:
    image: redis:alpine
    deploy:
      resources:
        limits:
          cpus: '0.001'
          memory: 50M
        reservations:
          cpus: '0.0001'
          memory: 20M
```

##### restart_policy

配置当容器退出时如何重启，会覆盖`restart`

- `condition`:`none`,`on-failure`,`any`(default)
- `delay`:重启尝试期间的等待时间,默认0
- `max_attempts`:最多尝试几次,默认为无限重试
- `window`:重启成功的决定时间为,默认为直接决定

```
version: "3"
services:
  redis:
    image: redis:alpine
    deploy:
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
        window: 120s
```

##### labels

为服务设置labels,而非任何服务内的容器

```
version: "3"
services:
  web:
    image: web
    deploy:
      labels:
        com.example.description: "This label will appear on the web service"
```

##### Not supported for docker stack deploy

以下在deploy时,不生效

- `build`
- `cgroup_parent`
- `container_name`
- `devices`
- `dns`
- `dns_search`
- `tmpfs`
- `external_links`
- `links`
- `network_mode`
- `security_opt`
- `stop_signal`
- `sysctls`
- `userns_mode`

#### devices

设备映射

```
devices:
  - "/dev/ttyUSB0:/dev/ttyUSB0"
```

#### depends_on

容器启动时根据依赖关系进行顺序启动,如下,首先启动redis和db,再启动web.

同时若`docker-compose up web`,则redis和db也会被启动  

只是保证启动顺序,而非保证在redis和db启动完成时web才启动.若需严格控制,则用另外一个方法[Controlling startup order in Compose | Docker Documentation](https://docs.docker.com/compose/startup-order/)

```
version: '3'
services:
  web:
    build: .
    depends_on:
      - db
      - redis
  redis:
    image: redis
  db:
    image: postgres
```

#### dns

```
dns: 8.8.8.8
dns:
  - 8.8.8.8
  - 9.9.9.9
```

#### dns_search

自定义dns搜索域

```
dns_search: example.com
dns_search:
  - dc1.example.com
  - dc2.example.com
```

#### tmpfs

在容器内挂载临时文件系统

```
tmpfs: /run
tmpfs:
  - /run
  - /tmp
```

#### entrypoint

重写默认的`entrypoint`,不仅覆盖dockerfile中的`ENTRYPOINT`,也使`CMD`失效

```
entrypoint: /code/entrypoint.sh
entrypoint:
    - php
    - -d
    - zend_extension=/usr/local/lib/php/extensions/no-debug-non-zts-20100525/xdebug.so
    - -d
    - memory_limit=-1
    - vendor/bin/phpunit
```

#### env_file

从文件中添加环境变量,该文件的目录为相对于`docker-compose.yml`的位置.这个环境变量会被`environment`中的值所覆盖.

```
env_file: .env

env_file:
  - ./common.env
  - ./apps/web.env
  - /opt/secrets.env
```

文件如下,右值若有引号,则引号会被加入到环境变量中去.若不同文件中含有相同的环境变量,则根据文件列表的从上到下,下面的会覆盖上面的.

```
# Set Rails/Rack environment
RACK_ENV=development
```

#### environment

`true, false, yes no`类的boolean值需要加引号,其他不加

```
environment:
  RACK_ENV: development
  SHOW: 'true'
  SESSION_SECRET:

environment:
  - RACK_ENV=development
  - SHOW=true
  - SESSION_SECRET
```

#### expose

开放端口,但是不对宿主机可见.仅仅对link过的容器可见.

```
expose:
 - "3000"
 - "8000"
```

#### external_links

与Compose外的容器进行连接,可以为各个容器提供公共服务.

```
external_links:
 - redis_1
 - project_db_1:mysql
 - project_db_1:postgresql
```

#### extra_hosts

添加主机名映射,类似于`--add-host`

```
extra_hosts:
 - "somehost:162.242.195.82"
 - "otherhost:50.31.209.229"
```

在容器中的`/etc/hosts`中会产生如下内容

```
162.242.195.82  somehost
50.31.209.229   otherhost
```

#### healthcheck

new in v2.1  运行一个检查,来判断该容器是否健康

```
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost"]
  interval: 1m30s
  timeout: 10s
  retries: 3
```

test写法,如果是列表,则第一项只能为`NONE`,`CMD`,`CMD-SHELL`,如果在字符串,则默认`CMD-SHELL`执行

```
# Hit the local web app
test: ["CMD", "curl", "-f", "http://localhost"]

# As above, but wrapped in /bin/sh. Both forms below are equivalent.
test: ["CMD-SHELL", "curl -f http://localhost && echo 'cool, it works'"]
test: curl -f https://localhost && echo 'cool, it works'
```

若要禁用Dockerfile中设置的健康检查,则可行的2种格式如下:

```
healthcheck:
  disable: true
healthcheck:
  test: ["NONE"]
```

#### image

指定容器的运行镜像,格式为`respository/tag`或者为部分IMAGE ID

```
image: redis
image: ubuntu:14.04
image: tutum/influxdb
image: example-registry.com:4000/postgresql
image: a4bc65fd
```

如果image不存在,若没有build,则会自动pull镜像,若有build,则会去build镜像.

#### isolation

指定容器的隔离技术,Linux下唯一支持的值为`defalut`,Windows下支持`default`,`process`,`hyperv`.

#### labels

指定容器的标签

```
labels:
  com.example.description: "Accounting webapp"
  com.example.department: "Finance"
  com.example.label-with-empty-value: ""

labels:
  - "com.example.description=Accounting webapp"
  - "com.example.department=Finance"
  - "com.example.label-with-empty-value"
```

#### links

指定了服务名称,或者服务名称:别名.这样容器中就可以通过别名来访问其他容器了.

```
web:
  links:
   - db
   - db:database
   - redis
```

#### logging

设置容器的log配置

```
logging:
  driver: syslog
  options:
    syslog-address: "tcp://192.168.0.42:123"
```

- `driver`:指定了日志设备,包括`json-file`(default),`syslog`,`none`

```
options:
  max-size: "200k"
  max-file: "10"
```

#### network_mode: "bridge"

```
network_mode: "host"
network_mode: "none"
network_mode: "service:[service name]"
network_mode: "container:[container name/id]"
```

#### networks

```
services:
  some-service:
    networks:
     - some-network
     - other-network
```

##### aliases

主机名的别名,网络内的容器既可以通过服务名访问,也可以通过别名访问.一个服务可以有很多别名.

```
services:
  some-service:
    networks:
      some-network:
        aliases:
         - alias1
         - alias3
      other-network:
        aliases:
         - alias2
```

```
version: '2'

services:
  web:
    build: ./web
    networks:
      - new

  worker:
    build: ./worker
    networks:
      - legacy

  db:
    image: mysql
    networks:
      new:
        aliases:
          - database
      legacy:
        aliases:
          - mysql

networks:
  new:
  legacy:
```

- `driver`:单主机为`bridge`,swarm为`overlay`
    - `bridge` [Introduction](https://github.com/docker/labs/blob/master/networking/A2-bridge-networking.md)
    - `overlay`[Introduction](https://github.com/docker/labs/blob/master/networking/A3-overlay-networking.md)
- `driver_opts`
- `enable_ipv6`
- `ipam`
    - `driver`:自定义IPAM驱动
    - `config`:0个或更多的配置块
        - `subnet`
- `internal`
- `labels`
- `external`

##### ipv4_address,ipv6_address

指定容器的静态ip地址

```
version: '2.1'

services:
  app:
    image: busybox
    command: ifconfig
    networks:
      app_net:
        ipv4_address: 172.16.238.10
        ipv6_address: 2001:3984:3989::10

networks:
  app_net:
    driver: bridge
    enable_ipv6: true
    ipam:
      driver: default
      config:
      -
        subnet: 172.16.238.0/24
      -
        subnet: 2001:3984:3989::/64
```

#### pid

```
pid: "host"
```

使得PID模式为主机PID模式,使得容器和操作系统之间通过PID 地址空间进行共享.

#### ports

```
ports:
 - "3000"
 - "3000-3005"
 - "8000:8000"
 - "9090-9091:8080-8081"
 - "49100:22"
 - "127.0.0.1:8001:8001"
 - "127.0.0.1:5000-5010:5000-5010"
 - "6060:6060/udp"
ports:
  - target: 80
    published: 8080
    protocol: tcp
    mode: host
```

long syntax 是v3.2新增的

- `target`:容器内端口
- `published`:主机端口
- `protocol`:`tcp`or`udp`
- `mode`:`host`or`ingress`

#### secrets

```
version: "3.1"
services:
  redis:
    image: redis:latest
    deploy:
      replicas: 1
    secrets:
      - my_secret
      - my_other_secret
secrets:
  my_secret:
    file: ./my_secret.txt
  my_other_secret:
    external: true
```

secrets文件将被挂载到容器中的`/run/secrets/<secret_name>`.other_secrets指的是通过其他方式定义的secrets,比如docker secrets create.

```
version: "3.1"
services:
  redis:
    image: redis:latest
    deploy:
      replicas: 1
    secrets:
      - source: my_secret
        target: redis_secret
        uid: '103'
        gid: '103'
        mode: 0440
secrets:
  my_secret:
    file: ./my_secret.txt
  my_other_secret:
    external: true
```

#### security_opt

```
security_opt:
  - label:user:USER
  - label:role:ROLE
```

#### stop_grace_period

```
stop_grace_period: 1s
stop_grace_period: 1m30s
```

默认10s

#### stop_signal

默认为SIGTERM

```
stop_signal: SIGUSR1
```

#### sysctls

容器内的内核参数

```
sysctls:
  net.core.somaxconn: 1024
  net.ipv4.tcp_syncookies: 0

sysctls:
  - net.core.somaxconn=1024
  - net.ipv4.tcp_syncookies=0
```

#### ulimits

覆盖容器内默认的ulimits

```
ulimits:
  nproc: 65535
  nofile:
    soft: 20000
    hard: 40000
```

#### userns_mode

```
userns_mode: "host"
```

#### volumes

```
volumes:
  # Just specify a path and let the Engine create a volume
  - /var/lib/mysql

  # Specify an absolute path mapping
  - /opt/data:/var/lib/mysql

  # Path on the host, relative to the Compose file
  - ./cache:/tmp/cache

  # User-relative path
  - ~/configs:/etc/configs/:ro

  # Named volume
  - datavolume:/var/lib/mysql
```

```
version: "3"

services:
  db:
    image: db
    volumes:
      - data-volume:/var/lib/db
  backup:
    image: backup-service
    volumes:
      - data-volume:/var/lib/backup/data

volumes:
  data-volume:
    external: true
    labels:
        com.example.description: "Database volume"
        com.example.department: "IT/Ops"
        com.example.label-with-empty-value: ""
    labels:
        - "com.example.description=Database volume"
        - "com.example.department=IT/Ops"
        - "com.example.label-with-empty-value"
```

- `driver`
- `driver-opts`
- `external`:设置为`true`,则标志该卷已经在Compose外创建,`docker-compose`不会创建之,若不存在,则会报错.
- `labels`

long syntax in v3.2

```
volumes:
  - type: volume
    source: mydata
    target: /data
    volume:
      nocopy: true
  - type: bind
    source: ./static
    target: /opt/app/static
```

#### restart

```
restart: "no"
restart: always
restart: on-failure
restart: unless-stopped
```

#### domainname, hostname, ipc, mac_address, privileged, read_only, shm_size, stdin_open, tty, user, working_dir

```
user: postgresql
working_dir: /code

domainname: foo.com
hostname: foo
ipc: host
mac_address: 02:42:ac:11:65:43

privileged: true


read_only: true
shm_size: 64M
stdin_open: true
tty: true
```
