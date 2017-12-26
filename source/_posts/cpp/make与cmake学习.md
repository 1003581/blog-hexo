---
title: make与cmake学习
date: 2017-09-14 16:00:18
tags: make
categories: c++
---

make与cmake
<!-- more -->
# make

## 简单的makefile

makefile规则如下：

```
target ... : prerequisites ...
  command
  ...
  ...
```

示例如下：

现有3个cpp，2个h文件

- printa.h
	```c++
	#pragma once

	void printA();
	```
- printa.cpp
	```c++
	#include <stdio.h>
	#include "printa.h"

	void printA() {
		printf("A");
	}
	```
- printb.h
	```c++
	#pragma once

	void printB();
	```
- printb.cpp
	```c++
	#include <stdio.h>
	#include "printb.h"

	void printB() {
		printf("B");
	}
	```
- printb.cpp
	```c++
	#include <stdio.h>
	#include "printa.h"
	#include "printb.h"

	int main() {
		printA();
		printB();
		return 0;
	}
	```

makefile如下

```
main: main.o printa.o printb.o
	g++ -o main main.o printa.o printb.o
printa.o: printa.h printa.cpp
	g++ -c printa.cpp
printb.o: printb.h printb.cpp
	g++ -c printb.cpp
main.o: main.cpp printa.h printb.h
	g++ -c main.cpp
clean:
	rm *.o main
```

执行make命令后，输出如下:

```
g++ -c main.cpp
g++ -c printa.cpp
g++ -c printb.cpp
g++ -o main main.o printa.o printb.o
```

执行ls

```
main  main.cpp  main.o  makefile  printa.cpp  printa.h  printa.o  printb.cpp  printb.h  printb.o
```

执行make clean

```
rm *.o main
```

## 通用makefile

如下 [github](https://github.com/liqiang311/other-code/blob/master/Makefile) ：

```makefile
CC = gcc
CXX = g++
CFLAGS = -O3 -march=native -fPIC -fomit-frame-pointer -Wall -pipe -minline-stringops-dynamically #-mfpmath=sse -ftracer
CXXFLAGS = $(CFLAGS)
CPPFLAGS = -Wp,-DNDEBUG,-D_FILE_OFFSET_BITS=64
ASFLAGS = -Wa,
LDFLAGS = -s -Wl,-O3

INCLDIR = -I. -I./include
LIBDIR = -L. -L./lib
LIBS = -lm -lz 
EXE = main
OBJS=\
	$(patsubst %.c,%.o,$(wildcard *.c)) \
	$(patsubst %.cpp,%.o,$(wildcard *.cpp)) \
	$(patsubst %.cxx,%.o,$(wildcard *.cxx)) \
	$(patsubst %.cc,%.o,$(wildcard *.cc))

.SUFFIXES: .c .cpp .cxx .cc .o
.c.o:
	$(CC) $(CPPFLAGS) $(CFLAGS) $(ASFLAGS) -c $*.c $(INCLDIR)
.cpp.o:
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(ASFLAGS) -c $*.cpp $(INCLDIR) 
.cxx.o:
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(ASFLAGS) -c $*.cxx $(INCLDIR)
.cc.o:
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(ASFLAGS) -c $*.cc $(INCLDIR)

all : $(EXE)

$(EXE) : $(OBJS)
	$(CXX) $(LDFLAGS) -o $(EXE) $(OBJS) $(LIBDIR) $(LIBS)

clean:
	rm -f $(EXE) $(OBJS)
```

若要生成.so，则要修改LDFLAGS，添加`shared`。详见[http://www.cppblog.com/deane/articles/165216.html](http://www.cppblog.com/deane/articles/165216.html)

# cmake

[cmake 学习笔记(一)](http://blog.csdn.net/dbzhang800/article/details/6314073)

## Demo

首先需要安装

```shell
apt-get install cmake
```

还是上文的几个文件

编写CMakeList.txt如下

```cmake
project(HELLO)
set(SRC_LIST main.cpp printa.cpp printb.cpp)
add_executable(hello ${SRC_LIST})
```

然后新建build文件夹，进入build文件夹执行以下命令

```
cmake ..
```

输出如下：

```
-- The C compiler identification is GNU 4.8.4
-- The CXX compiler identification is GNU 4.8.4
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Configuring done
-- Generating done
-- Build files have been written to: /root/lq/build
```

然后make

```
Scanning dependencies of target hello
[ 33%] Building CXX object CMakeFiles/hello.dir/main.cpp.o
[ 66%] Building CXX object CMakeFiles/hello.dir/printa.cpp.o
[100%] Building CXX object CMakeFiles/hello.dir/printb.cpp.o
Linking CXX executable hello
[100%] Built target hello
```

于是生成了hello可执行文件。

## 基础模版

基础模版如下

```cmake
# CMake 最低版本号要求
cmake_minimum_required (VERSION 2.8)

# 设置变量,如DemoTest
set(PROJECT_NAME DemoTest)  #项目名称
set(OBJ_NAME Demo)  #生成的目标文件名称
set(SUBDIR_SRCS ${PROJECT_SOURCE_DIR}/math) #文件子目录
set(LIB_DIR ${PROJECT_SOURCE_DIR}/lib) #链接库目录
set(LIB_FILE MathFunctions) #链接库文件

# 项目信息
project (${PROJECT_NAME})

#include_directories ("${PROJECT_SOURCE_DIR}/math")
#add_subdirectory (math)  
#set (EXTRA_LIBS ${EXTRA_LIBS} MathFunctions)

# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_SRCS 变量
aux_source_directory(. DIR_SRCS)

#开启gdb调试,debug模式下关闭优化,并打印警告,加调试选项
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

#添加C++11支持及其他选项 
set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS} -g -ftest-coverage -fprofile-arcs -Wno-deprecated")
MESSAGE(STATUS,"C++ flags have ",${CMAKE_CXX_FLAGS})

#指定添加的库文件
link_libraries(${EXTRA_LIBS})

# 指定生成目标
add_executable(${OBJ_NAME} ${DIR_SRCS})
target_link_libraries (${OBJ_NAME})

# 指定安装路径
install (TARGETS ${OBJ_NAME} DESTINATION bin)
```

## CMakeLists.txt语法

[CMake之CMakeLists.txt编写入门](http://blog.csdn.net/z_h_s/article/details/50699905)

# 两者区别

[CMake和Make之间的区别](http://blog.csdn.net/android_ruben/article/details/51698498)
