---
title: G++编译C++程序
date: 2017-09-14 15:58:29
tags: 
- g++
- c++
categories: c++
---

# C++ 编程中相关文件后缀

<!-- more -->

- 静态库 (archive) `.a`
- C++源代码（需要编译预处理）`.C` `.c` `.cc` `.cp` `.cpp` `.cxx` `.c++`
- C或者C++源代码头文件 `.h`
- C++源代码（不需编译预处理） `.ii`
- 对象文件 `.o`
- 汇编语言代码 `.s`
- 动态库 `.so`

# 单个源文件生成可执行程序

下面是一个保存在文件 `helloworld.cpp` 中一个简单的 C++ 程序的代码：

```c++
/* helloworld.cpp */
#include int main(int argc,char*argv[]){
    std::cout << "hello, world" << std::endl;
    return(0);
}
```

程序使用定义在头文件 `iostream` 中的 `cout`，向标准输出写入一个简单的字符串。该代码可用以下命令编译为可执行文件：

```bash
g++ helloworld.cpp
```

编译器 `g++` 通过检查命令行中指定的文件的后缀名可识别其为 C++ 源代码文件。

编译器默认的动作：编译源代码文件生成对象文件(object file)，链接对象文件和 libstdc++ 库中的函数得到可执行程序。然后删除对象文件。由于命令行中未指定可执行程序的文件名，编译器采用默认的 a.out。程序可以这样来运行：

```bash
./a.out
hello, world
```

更普遍的做法是通过 `-o` 选项指定可执行程序的文件名。下面的命令将产生名为 helloworld 的可执行文件：

```bash
g++ helloworld.cpp -o helloworld
```

在命令行中输入程序名可使之运行：

```bash
./helloworld
hello, world
```

程序 g++ 是将 gcc 默认语言设为 C++ 的一个特殊的版本，链接时它自动使用 C++ 标准库而不用 C 标准库。通过遵循源码的命名规范并指定对应库的名字，用 gcc 来编译链接 C++ 程序是可行的，如下例所示：

```bash
gcc helloworld.cpp -lstdc++ -o helloworld
```

选项 -l (ell) 通过添加前缀 lib 和后缀 .a 将跟随它的名字变换为库的名字 `libstdc++.a`。而后它在标准库路径中查找该库。

gcc 的编译过程和输出文件与 g++ 是完全相同的。

在大多数系统中，GCC 安装时会安装一名为 c++ 的程序。如果被安装，它和 g++ 是等同，如下例所示，用法也一致：

```bash
c++ helloworld.cpp -o helloworld
```

# 多个源文件生成可执行程序

如果多于一个的源码文件在 g++ 命令中指定，它们都将被编译并被链接成一个单一的可执行文件。

下面是一个名为 `speak.h` 的头文件；它包含一个仅含有一个函数的类的定义：

```c++
/* speak.h */
#include class Speak{
    public:void sayHello(constchar*);
}
```

下面列出的是文件 `speak.cpp` 的内容：包含 `sayHello()`` 函数的函数体：

```c++
/* speak.cpp */
#include "speak.h"
void Speak::sayHello(constchar*str){
    std::cout << "Hello " << str << "\n";
}
```

文件 `hellospeak.cpp` 内是一个使用 Speak 类的程序：

```c++
/* hellospeak.cpp */
#include "speak.h"
int main(int argc,char*argv[]){
    Speak speak;
    speak.sayHello("world");
    return(0);
}
```

下面这条命令将上述两个源码文件编译链接成一个单一的可执行程序：

```bash
g++ hellospeak.cpp speak.cpp -o hellospeak
```

PS：这里说一下为什么在命令中没有提到“speak.h“该文件

原因是：在`speak.cpp`中包含有`#include"speak.h"`这句代码，它的意思是搜索系统头文件目录之前将先在当前目录中搜索文件`speak.h`。而`speak.h`正在该目录中，不用再在命令中指定了。

# 源文件生成对象文件

选项 `-c` 用来告诉编译器编译源代码但不要执行链接，输出结果为对象文件。文件默认名与源码文件名相同，只是将其后缀变为 `.o`。

例如，下面的命令将编译源码文件 `hellospeak.cpp` 并生成对象文件 `hellospeak.o`：

```bash
g++ -c hellospeak.cpp
```

命令 g++ 也能识别 `.o` 文件并将其作为输入文件传递给链接器。下列命令将编译源码文件为对象文件并将其链接成单一的可执行程序：

```bash
g++ -c hellospeak.cpp
g++ -c speak.cpp
g++ hellospeak.o speak.o -o hellospeak
```

选项 `-o` 不仅仅能用来命名可执行文件。它也用来命名编译器输出的其他文件。例如：除了中间的对象文件有不同的名字外，下列命令将生成和上面完全相同的可执行文件：

```bash
g++ -c hellospeak.cpp -o hspk1.o 
g++ -c speak.cpp -o hspk2.o 
g++ hspk1.o hspk2.o -o hellospeak
```

# 编译预处理

选项 `-E` 使 g++ 将源代码用编译预处理器处理后不再执行其他动作。

下面的命令预处理源码文件 `helloworld.cpp` 并将结果显示在标准输出中：

```bash
g++ -E helloworld.cpp
```

本文前面所列出的 `helloworld.cpp` 的源代码，仅仅有六行，而且该程序除了显示一行文字外什么都不做，但是，预处理后的版本将超过 1200 行。这主要是因为头文件 `iostream` 被包含进来，而且它又包含了其他的头文件，除此之外，还有若干个处理输入和输出的类的定义。

预处理过的文件的 GCC 后缀为 `.ii`，它可以通过 `-o` 选项来生成，例如：

```bash
gcc -E helloworld.cpp -o helloworld.ii
```

# 生成汇编代码

选项 `-S` 指示编译器将程序编译成汇编语言，输出汇编语言代码而后结束。

下面的命令将由 C++ 源码文件生成汇编语言文件 `helloworld.s`：

```bash
g++ -S helloworld.cpp
```

生成的汇编语言依赖于编译器的目标平台。

# 创建静态库

静态库是编译器生成的一系列对象文件的集合。

链接一个程序时用库中的对象文件还是目录中的对象文件都是一样的。库中的成员包括普通函数，类定义，类的对象实例等等。静态库的另一个名字叫归档文件(archive)，管理这种归档文件的工具叫 `ar` 。

在下面的例子中，我们先创建两个对象模块，然后用其生成静态库。

头文件 `say.h` 包含函数 `sayHello()` 的原型和类 Say 的定义：

```c++
/* say.h */
#include void sayhello(void);
class Say {
private:
    char*string;
public:
    Say(char*str){
        string= str;
    }
    void sayThis(constchar*str){
        std::cout << str << " from a static library\n";
    }
    void sayString(void);
};
```

下面是文件 `say.cpp` 是我们要加入到静态库中的两个对象文件之一的源码。它包含 Say 类中 `sayString()` 函数的定义体；类 Say 的一个实例 librarysay 的声明也包含在内：

```c++
/* say.cpp */
#include "say.h"
void Say::sayString(){
    std::cout << string << "\n";
}
Say librarysay("Library instance of Say");
```

源码文件 `sayhello.cpp` 是我们要加入到静态库中的第二个对象文件的源码。它包含函数 `sayhello()` 的定义：

```c++
/* sayhello.cpp */
#include "say.h"
void sayhello(){
    std::cout << "hello from a static library\n";
}
```

下面的命令序列将源码文件编译成对象文件，命令 `ar` 将其存进库中：

```bash
g++ -c sayhello.cpp
g++ -c say.cpp
ar -r libsay.a sayhello.o say.o
```

程序 `ar` 配合参数 `-r` 创建一个新库 `libsay.a` 并将命令行中列出的对象文件插入。采用这种方法，如果库不存在的话，参数 `-r` 将创建一个新的库，而如果库存在的话，将用新的模块替换原来的模块。

下面是主程序 `saymain.cpp`，它调用库 `libsay.a` 中的代码：

```c++
/* saymain.cpp */
#include "say.h"
int main(int argc,char*argv[]){
    extern Say librarysay;
    Say localsay = Say("Local instance of Say");
    sayhello();
    librarysay.sayThis("howdy");
    librarysay.sayString();
    localsay.sayString();
    return(0);
}
```

该程序可以下面的命令来编译和链接：

```bash
g++ saymain.cpp libsay.a -o saymain
```

程序运行时，产生以下输出：

```
hello from a static library
howdy from a static library
Library instance of Say
Local instance of Say
```

本文来自： http://wiki.ubuntu.org.cn/Compiling_Cpp 

ps：如果一个文件夹下有多个cpp文件需要编译的话，除了采用makefile的方式之外，还可以使用

```bash
g++ *.cpp -o hello
```

hello为编译生成的可执行文件的名字，编译时要确保cpp文件和他们各自所引用的头文件在同一个目录下。
