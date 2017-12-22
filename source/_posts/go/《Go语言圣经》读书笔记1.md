---
title: 《Go语言圣经》读书笔记1
date: 2017-10-19 09:04
tags: go
categories: go
---

## 链接

- [GitHub](https://github.com/golang-china/gopl-zh)
- [读书地址](https://docs.hacknode.org/gopl-zh/)
- [另一本书《the-way-to-go》](https://github.com/Unknwon/the-way-to-go_ZH_CN)
<!-- more -->

## 相关链接

- [Packages - The Go Programming Language](https://golang.org/pkg/)
- [http - The Go Programming Language](https://golang.org/pkg/net/http/)
- [使用Go构建RESTful的JSON API - 酱油蔡的酱油坛 - CSDN博客](http://blog.csdn.net/xingwangc2014/article/details/51623157)
- [编写HTTP客户端 · Golang Web](http://dmdgeeker.com/goBook/docs/ch03/client.html)
- [Go语言版crontab | Go语言中文网博客](http://blog.studygolang.com/2014/02/go_crontab/)
- [go - Golang - Getting a slice of keys from a map - Stack Overflow](https://stackoverflow.com/questions/21362950/golang-getting-a-slice-of-keys-from-a-map)
- [syncmap - GoDoc](https://godoc.org/golang.org/x/sync/syncmap)
- [go - ERROR: need type assertion - Stack Overflow](https://stackoverflow.com/questions/40683635/error-need-type-assertion)
- [The-Golang-Standard-Library-by-Example/02.3.md at master · polaris1119/The-Golang-Standard-Library-by-Example](https://github.com/polaris1119/The-Golang-Standard-Library-by-Example/blob/master/chapter02/02.3.md)
- [golang时间处理 - Go语言中文网 - Golang中文社区](https://studygolang.com/articles/736)
- [Go实现比较时间大小_Golang_脚本之家](http://m.jb51.net/article/64705.htm)
- [Go 语言 JSON 简介 – Cizixs Writes Here](http://cizixs.com/2016/12/19/golang-json-guide)
- [【GoLang笔记】遍历map时的key随机化问题及解决方法 - Go语言中文网 - Golang中文社区](https://studygolang.com/articles/2749)
- [golang中时间(time)的相关操作 - 快乐编程](http://www.01happy.com/golang-time/)
- [[Go语言]你真的了解fmt.Printlf和fmt.Println的区别？_stevewang_新浪博客](http://blog.sina.com.cn/s/blog_9be3b8f10101e6oe.html)
- [strconv - The Go Programming Language](https://golang.org/pkg/strconv/)
- [类型转换 - golang []interface{} 的数组如何转换为 []string 的数组 - SegmentFault](https://segmentfault.com/q/1010000003505053)
- [go 语言获取文件名、后缀 - woquNOKIA的专栏 - CSDN博客](http://webcache.googleusercontent.com/search?q=cache:irMcZlR16XQJ:blog.csdn.net/woquNOKIA/article/details/50052219+&cd=2&hl=zh-CN&ct=clnk&gl=hk&client=aff-cs-360chromium)


## 前言

### Go与C++的比较

技术选型

- Python+Redis 脚本语言开发迅速、数据结构丰富、也有多线程等模式 无法处理复杂数据结构、需要做C扩展或者使用很多复杂的实现方式，常驻内存后端程序还是不够稳定，无法处理非常高性能的场合，也不方便优化 
- C++ +Redis 高性能、极强的定制性、可以实现任何内存和数据结构、可以达到任何功能和性能的要求和强大的底层操控能力 开发维护成本高、出现问题几率较大、没有专职C++开发人员，需要构建很多基础库、包括异步网络框架，采用多线程还是异步IO、配置文件解析、内存管理、日志等基础库的开发 
- Go+Redis 学习曲线短、高性能、高并发，支持强大的的类C的内存和各种数据结构操作可以满足一般场景需求

`=` 和 `:= `

- C++中没有`:=`这个符号
- Go语言中：=用来声明一个变量的同时给这个变量赋值 并且只能在函数体内使用
- 主要是为了省略类型声明，系统会自己选择相应的类型识别定义的变量

New和make操作符

- C++中new操作符用于给一个对象 分配内存但是没有清零
- go语言中调用new（T）被求值时 所做的是为T类型的新值分配并且清零一块内存空间，然后将内存空间的地址作为结果返回 所以不用担心乱码问题 可以直接拿来使用
- make用于内建类型（map、slice 和channel）的内存分配。new用于各种类型的内存分配。 
- make只能创建slice、map和channel，并且返回一个有初始值(非零)的T类型（引用），而不是*T。

面向对象

- go语言和c++在面向对象方面完全没有可比性，尤其是没有继承、泛型、虚函数、函数重载、构造函数、析构函数等。

## 入门

Go语言编译过程没有警告信息，争议特性之一

在表达式x + y中，可在+后换行，不能在+前换行（译注：以+结尾的话不会被插入分号分隔符，但是以x结尾的话则会被分号分隔符，从而导致编译错误）

根据代码需要, 自动地添加或删除import声明 `go get golang.org/x/tools/cmd/goimports`

### 命令行参数

```go
package main

import (
	"fmt"
	"os"
	"strings"
)

func main() {
	var s, sep string
	for _, args := range os.Args {
		s += sep + args
		sep = " "
	}
	fmt.Println(s)

	fmt.Println(strings.Join(os.Args, " "))
}
```

### 查找重复的行

```
%d          十进制整数
%x, %o, %b  十六进制，八进制，二进制整数。
%f, %g, %e  浮点数： 3.141593 3.141592653589793 3.141593e+00
%t          布尔：true或false
%c          字符（rune） (Unicode码点)
%s          字符串
%q          带双引号的字符串"abc"或带单引号的字符'c'
%v          变量的自然形式（natural format）
%T          变量的类型
%%          字面上的百分号标志（无操作数）
```

```go
package main

import (
	"bufio"
	"fmt"
	"os"
)

func main() {
	counts := make(map[string]int)
	files := os.Args[1:]
	if len(files) == 0 {
		countLines(os.Stdin, counts)
	} else {
		for _, arg := range files {
			f, err := os.Open(arg)
			if err != nil {
				fmt.Fprintf(os.Stderr, "dup2: %v\n", err)
				continue
			}
			countLines(f, counts)
			f.Close()
		}
	}
	for line, n := range counts {
		if n > 1 {
			fmt.Printf("%d\t%s\n", n, line)
		}
	}
}

func countLines(f *os.File, counts map[string]int) {
	input := bufio.NewScanner(f)
	for input.Scan() {
		counts[input.Text()]++
	}
	// NOTE: ignoring potential errors from input.Err()
}
```

### GIF动画

```go
package main

import (
	"fmt"
	"image"
	"image/color"
	"image/gif"
	"io"
	"math"
	"math/rand"
	"os"
	"time"
)

//var palette = []color.Color{color.White, color.Black}
var palette = []color.Color{color.RGBA{0x00, 0xff, 0x00, 0xff}, color.RGBA{0xff, 0x00, 0x00, 0xff}}

const (
	backgroudIndex = 0 // first color in palette
	foregroudIndex = 1 // next color in palette
)

func main() {
	// The sequence of images is deterministic unless we seed
	// the pseudo-random number generator using the current time.
	// Thanks to Randall McPherson for pointing out the omission.
	rand.Seed(time.Now().UTC().UnixNano())
	fout, err := os.Create("out.gif")
	defer fout.Close()
	if err != nil {
		fmt.Println("out.gif", err)
		return
	}
	//lissajous(os.Stdout)
	lissajous(fout)
}

func lissajous(out io.Writer) {
	const (
		cycles  = 5     // number of complete x oscillator revolutions
		res     = 0.001 // angular resolution
		size    = 100   // image canvas covers [-size..+size]
		nframes = 64    // number of animation frames
		delay   = 8     // delay between frames in 10ms units
	)

	freq := rand.Float64() * 3.0 // relative frequency of y oscillator
	anim := gif.GIF{LoopCount: nframes}
	phase := 0.0 // phase difference
	for i := 0; i < nframes; i++ {
		rect := image.Rect(0, 0, 2*size+1, 2*size+1)
		img := image.NewPaletted(rect, palette)
		for t := 0.0; t < cycles*2*math.Pi; t += res {
			x := math.Sin(t)
			y := math.Sin(t*freq + phase)
			img.SetColorIndex(size+int(x*size+0.5), size+int(y*size+0.5),
				foregroudIndex)
		}
		phase += 0.1
		anim.Delay = append(anim.Delay, delay)
		anim.Image = append(anim.Image, img)
	}
	gif.EncodeAll(out, &anim) // NOTE: ignoring encoding errors
}
```

### 并发获取多个URL

```go
package main

import (
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"strings"
	"time"
)

func main() {
	start := time.Now()
	ch := make(chan string)
	for _, url := range os.Args[1:] {
		go fetch(url, ch) // start a goroutine
	}
	for range os.Args[1:] {
		fmt.Println(<-ch) // receive from channel ch
	}
	fmt.Printf("%.2fs elapsed\n", time.Since(start).Seconds())
}

func fetch(url string, ch chan<- string) {
	start := time.Now()
	if ok := strings.HasPrefix(url, "http"); !ok {
		url = "http://" + url
	}
	resp, err := http.Get(url)
	if err != nil {
		ch <- fmt.Sprint(err) // send to channel ch
		return
	}

	nbytes, err := io.Copy(ioutil.Discard, resp.Body)
	resp.Body.Close() // don't leak resources
	if err != nil {
		ch <- fmt.Sprintf("while reading %s: %v", url, err)
		return
	}
	secs := time.Since(start).Seconds()
	ch <- fmt.Sprintf("%.2fs  %7d  %s", secs, nbytes, url)
}
```

### Web服务

```go
package main

import (
	"fmt"
	"image"
	"image/color"
	"image/gif"
	"io"
	"log"
	"math"
	"math/rand"
	"net/http"
	"strconv"
	"sync"
)

var mu sync.Mutex
var count int

func main() {
	http.HandleFunc("/", handler)
	http.HandleFunc("/gif", handlerGif)
	http.HandleFunc("/count", counter)
	log.Fatal(http.ListenAndServe("localhost:8000", nil))
}

// handler echoes the Path component of the requested URL.
func handler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "%s %s %s\n", r.Method, r.URL, r.Proto)
	for k, v := range r.Header {
		fmt.Fprintf(w, "Header[%q] = %q\n", k, v)
	}
	fmt.Fprintf(w, "Host = %q\n", r.Host)
	fmt.Fprintf(w, "RemoteAddr = %q\n", r.RemoteAddr)
	if err := r.ParseForm(); err != nil {
		log.Print(err)
	}
	for k, v := range r.Form {
		fmt.Fprintf(w, "Form[%q] = %q\n", k, v)
	}
}

// counter echoes the number of calls so far.
func counter(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	fmt.Fprintf(w, "Count %d\n", count)
	mu.Unlock()
}

func handlerGif(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		log.Print(err)
	}
	cycles, _ := strconv.ParseFloat(r.Form.Get("cycles"), 10)
	log.Println(cycles)
	lissajous(w, cycles)
}

//var palette = []color.Color{color.White, color.Black}
var palette = []color.Color{color.RGBA{0x00, 0xff, 0x00, 0xff}, color.RGBA{0xff, 0x00, 0x00, 0xff}}

const (
	backgroudIndex = 0 // first color in palette
	foregroudIndex = 1 // next color in palette
)

func lissajous(out io.Writer, cycles float64) {
	const (
		//cycles  = 5     // number of complete x oscillator revolutions
		res     = 0.001 // angular resolution
		size    = 100   // image canvas covers [-size..+size]
		nframes = 64    // number of animation frames
		delay   = 8     // delay between frames in 10ms units
	)

	freq := rand.Float64() * 3.0 // relative frequency of y oscillator
	anim := gif.GIF{LoopCount: nframes}
	phase := 0.0 // phase difference
	for i := 0; i < nframes; i++ {
		rect := image.Rect(0, 0, 2*size+1, 2*size+1)
		img := image.NewPaletted(rect, palette)
		for t := 0.0; t < cycles*2*math.Pi; t += res {
			x := math.Sin(t)
			y := math.Sin(t*freq + phase)
			img.SetColorIndex(size+int(x*size+0.5), size+int(y*size+0.5),
				foregroudIndex)
		}
		phase += 0.1
		anim.Delay = append(anim.Delay, delay)
		anim.Image = append(anim.Image, img)
	}
	gif.EncodeAll(out, &anim) // NOTE: ignoring encoding errors
}
```

### Switch

tag switch(tagless switch) ---- Go语言里的switch还可以不带操作对象（译注：switch不带操作对象时默认用true值代替，然后将每个case的表达式和true值进行比较）

## 程序结构

### 命名

关键字

```
break      default       func     interface   select
case       defer         go       map         struct
chan       else          goto     package     switch
const      fallthrough   if       range       type
continue   for           import   return      var
```

预先定义的名字并不是关键字，你可以在定义中重新使用它们

```
内建常量: true false iota nil

内建类型: int int8 int16 int32 int64
          uint uint8 uint16 uint32 uint64 uintptr
          float32 float64 complex128 complex64
          bool byte rune string error

内建函数: make len cap new append copy close delete
          complex real imag
          panic recover
```

在习惯上，Go语言程序员推荐使用 驼峰式 命名

### 声明

Go语言主要有四种类型的声明语句：var、const、type和func，分别对应变量、常量、类型和函数实体对象的声明。

### 变量

Go语言中不存在未初始化的变量。

在包级别声明的变量会在main入口函数执行前完成初始化（§2.6.2），局部变量将在声明语句被执行到的时候完成初始化。

变量的生命周期

- 包一级声明的变量 和整个程序的运行周期是一致的
- 局部变量的生命周期则是动态的：每次从创建一个新变量的声明语句开始，直到该变量不再被引用为止，然后变量的存储空间可能被回收。
    - 函数的参数变量和返回值变量都是局部变量。它们在函数每次被调用的时候创建。

垃圾回收

- 实现思路是，从每个包级的变量和每个当前运行函数的每一个局部变量开始，通过指针或引用的访问路径遍历，是否可以找到该变量。如果不存在这样的访问路径，那么说明该变量是不可达的，也就是说它是否存在并不会影响程序后续的计算结果。
- 因为一个变量的有效周期只取决于是否可达，因此一个循环迭代内部的局部变量的生命周期可能超出其局部作用域。同时，局部变量可能在函数返回之后依然存在。
- 编译器会自动选择在栈上还是在堆上分配局部变量的存储空间，但可能令人惊讶的是，这个选择并不是由用var还是new声明变量的方式决定的。
    ```go
    var global *int

    func f() {
        var x int
        x = 1
        global = &x
    }

    func g() {
        y := new(int)
        *y = 1
    }
    ```
- f函数里的x变量必须在堆上分配，因为它在函数退出后依然可以通过包一级的global变量找到，虽然它是在函数内部定义的；用Go语言的术语说，这个x局部变量从函数f中逃逸了。相反，当g函数返回时，变量`*y`将是不可达的，也就是说可以马上被回收的。因此，`*y`并没有从函数g中逃逸，编译器可以选择在栈上分配`*y`的存储空间（译注：也可以选择在堆上分配，然后由Go语言的GC回收这个变量的内存空间），虽然这里用的是new方式。

### 赋值

自增和自减是语句，而不是表达式，因此x = i++之类的表达式是错误的）

元组赋值
```
x, y = y, x
```

### 类型

```go
type Celsius float64    // 摄氏温度
type Fahrenheit float64 // 华氏温度
```
- Celsius和Fahrenheit分别对应不同的温度单位。它们虽然有着相同的底层类型float64，但是它们是不同的数据类型，因此它们不可以被相互比较或混在一个表达式运算。
- 因此需要一个类似Celsius(t)或Fahrenheit(t)形式的显式转型操作才能将float64转为对应的类型。

### 包和文件

因为汉字不区分大小写，因此汉字开头的名字是没有导出的

Import

- 按照惯例，一个包的名字和包的导入路径的最后一个字段相同，例如`gopl.io/ch2/tempconv`包的名字一般是tempconv。
- import别名
    ```
    import (
        sort "wuxu.bit/example/alg/sort"
        "fmt"
    )
    ```

golang在引入包的时候还有一些tricks：

- `import . "fmt"` 这样在调用fmt包的导出方法时可以省略fmt
- `import _ "fmt"` 这样引入该包但是不引入该包的导出函数，而是为了使用该导入操作的副作用: 调用包里面的init函数

包的初始化

- 包的初始化首先是解决包级变量的依赖顺序，然后按照包级变量声明出现的顺序依次初始化
- 初始化工作是自下而上进行的，main包最后被初始化。以这种方式，可以确保在main函数执行之前，所有依赖的包都已经完成初始化工作了。(包括init函数)
 
### 作用域

if,switch条件部分为一个隐式词法域，然后是每个分支的词法域。

## 基础数据类型

Go语言将数据类型分为四类：基础类型、复合类型、引用类型和接口类型。

本章介绍基础类型，包括：数字、字符串和布尔型。

### 整型

等价类型

- Unicode字符rune类型是和int32等价的类型,通常用于表示一个Unicode码点。
- byte也是uint8类型的等价类型，byte类型一般用于强调数值是一个原始的数据而不是一个小的整数。

二元运算符，它们按照优先级递减的顺序排列如下，二元运算符有五种优先级。在同一个优先级，使用左优先结合规则，但是使用括号可以明确优先顺序，使用括号也可以用于提升优先级

```
*      /      %      <<       >>     &       &^
+      -      |      ^
==     !=     <      <=       >      >=
&&
||
```

位操作符

```
&      位运算 AND
|      位运算 OR
^      位运算 XOR
&^     位清空 (AND NOT)
<<     左移
>>     右移
```

位操作运算符^作为二元运算符时是按位异或（XOR），当用作一元运算符时表示按位取反；

位操作代码

```go
package main

import (
	"fmt"
)

func main() {
	var x uint8 = 1<<1 | 1<<5
	var y uint8 = 1<<1 | 1<<2

	fmt.Printf("%08b\n", x) // "00100010", the set {1, 5}
	fmt.Printf("%08b\n", y) // "00000110", the set {1, 2}

	fmt.Printf("%08b\n", x&y)  // "00000010", the intersection {1}
	fmt.Printf("%08b\n", x|y)  // "00100110", the union {1, 2, 5}
	fmt.Printf("%08b\n", x^y)  // "00100100", the symmetric difference {2, 5}
	fmt.Printf("%08b\n", x&^y) // "00100000", the difference {5}

	for i := uint(0); i < 8; i++ {
		if x&(1<<i) != 0 { // membership test
			fmt.Println(i) // "1", "5"
		}
	}

	fmt.Printf("%08b\n", x<<1) // "01000100", the set {2, 6}
	fmt.Printf("%08b\n", x>>1) // "00010001", the set {0, 4}
}
```

对于一个`float`，可以利用`int(f)`这样的语法进行转化，但是会丢弃小数部分

打印八进制、十进制、十六进制

```go
o := 0666
fmt.Printf("%d %[1]o %#[1]o\n", o) // "438 666 0666"
x := int64(0xdeadbeef)
fmt.Printf("%d %[1]x %#[1]x %#[1]X\n", x)
// Output:
// 3735928559 deadbeef 0xdeadbeef 0XDEADBEEF
```

- %之后的[1]副词告诉Printf函数再次使用第一个操作数。
- %后的#副词告诉Printf在用%o、%x或%X输出时生成0、0x或0X前缀。

### 浮点数

一个float32类型的浮点数可以提供大约6个十进制数的精度，而float64则可以提供约15个十进制数的精度；

### 复数

Go语言提供了两种精度的复数类型：complex64和complex128，分别对应float32和float64两种浮点数精度。内置的complex函数用于构建复数，内建的real和imag函数分别返回复数的实部和虚部：

```go
var x complex128 = complex(1, 2) // 1+2i
var y complex128 = complex(3, 4) // 3+4i
fmt.Println(x*y)                 // "(-5+10i)"
fmt.Println(real(x*y))           // "-5"
fmt.Println(imag(x*y))           // "10"
```

### 布尔型

```go
func btoi(b bool) int {
    if b {
        return 1
    }
    return 0
}
func itob(i int) bool { return i != 0 }
```

### 字符串

内置的len函数可以返回一个字符串中的字节数目（不是rune字符数目）

字符串可以用==和<进行比较；比较通过逐个字节比较完成的，因此比较的结果是字符串自然编码的顺序。

字符串的值是不可变的：

原生的字符串面值`\`...\``，用\`包含的字符串中没有转义操作

Unicode和UTF-8

- Unicode每个符号都分配一个唯一的Unicode码点，Unicode码点对应Go语言中的rune整数类型（译注：rune是int32等价类型）。
- UTF8编码使用1到4个字节来表示每个Unicode码点，ASCII部分字符只使用1个字节，常用字符部分使用2或3个字节表示。

```
0xxxxxxx                             runes 0-127    (ASCII)
110xxxxx 10xxxxxx                    128-2047       (values <128 unused)
1110xxxx 10xxxxxx 10xxxxxx           2048-65535     (values <2048 unused)
11110xxx 10xxxxxx 10xxxxxx 10xxxxxx  65536-0x10ffff (other values unused)
```

统计utf8的字符串中rune的个数，使用`utf8.RuneCountInString(s)`

转换代码

```go
// "program" in Japanese katakana
s := "プログラム"
fmt.Printf("% x\n", s) // "e3 83 97 e3 83 ad e3 82 b0 e3 83 a9 e3 83 a0"
r := []rune(s)
fmt.Printf("%x\n", r)  // "[30d7 30ed 30b0 30e9 30e0]"
fmt.Println(string(r)) // "プログラム"
fmt.Println(string(65))     // "A", not "65"
fmt.Println(string(0x4eac)) // "京"
fmt.Println(string(1234567)) // "?"
```
> 在第一个Printf中的% x参数用于在每个十六进制数字前插入一个空格。

bytes、strings、strconv和unicode

- bytes.Buffer提供构建字符串。
- strconv包提供了布尔型、整型数、浮点数和对应字符串的相互转换，还提供了双引号转义相关的转换。
- unicode包提供了IsDigit、IsLetter、IsUpper和IsLower等类似功能，它们用于给字符分类。每个函数有一个单一的rune类型的参数，然后返回一个布尔值。

### 常量

- 常量表达式的值在编译期计算，而不是在运行期。每种常量的潜在类型都是基础类型：boolean、string或数字。
- 常量间的所有算术运算、逻辑运算和比较运算的结果也是常量，对常量的类型转换操作或以下函数调用都是返回常量结果：len、cap、real、imag、complex和unsafe.Sizeof

iota 常量生成器

```go
const (
    FlagUp Flags = 1 << iota // is up
    FlagBroadcast            // supports broadcast access capability
    FlagLoopback             // is a loopback interface
    FlagPointToPoint         // belongs to a point-to-point link
    FlagMulticast            // supports multicast access capability
)
const (
    _ = 1 << (10 * iota)
    KiB // 1024
    MiB // 1048576
    GiB // 1073741824
    TiB // 1099511627776             (exceeds 1 << 32)
    PiB // 1125899906842624
    EiB // 1152921504606846976
    ZiB // 1180591620717411303424    (exceeds 1 << 64)
    YiB // 1208925819614629174706176
)
```

## 复合数据类型

数组、slice、map和结构体

- 数组和结构体是聚合类型；它们的值由许多元素或成员字段的值组成。
- 数组是由同构的元素组成——每个数组元素都是完全相同的类型——结构体则是由异构的元素组成的。
- 数组和结构体都是有固定内存大小的数据结构。
- slice和map则是动态的数据结构，它们将根据需要动态增长。

### 数组

长度固定

初始化

```go
var a [3]int
var b [3]int = [3]int{1, 2, 3}
var c [3]int = [3]int{1, 2}
d := [...]int{1, 2, 3} //数组的长度是根据初始化值的个数来计算
e := [...]int{99: -1} //定义了一个含有100个元素的数组r，最后一个元素被初始化为-1，其它元素都是用0初始化。
```

数组比较

- 相同类型（相同长度）的才能比较
- 不同类型比较会编译错误
- 数组中的元素完全一样，才是相等

```go
a := [2]int{1, 2}
b := [...]int{1, 2}
c := [2]int{1, 3}
fmt.Println(a == b, a == c, b == c) // "true false false"
d := [3]int{1, 2}
fmt.Println(a == d) // compile error: cannot compare [2]int == [3]int
```

函数传参

- 函数参数变量接收的是一个复制的副本
- 传递大的数组类型将是低效的
- Go语言对待数组的方式和其它很多编程语言不同，其它编程语言可能会隐式地将数组作为引用或指针对象传入被调用的函数。
- 改进：传递数组指针
```go
func zero(ptr *[32]byte) {
}
```

sha256创建及比较

```go
package main

import (
	"crypto/sha256"
	"fmt"
)

var pc [256]byte

func init() {
	for i := range pc {
		pc[i] = pc[i/2] + byte(i&1)
	}
}

func main() {
	c1 := sha256.Sum256([]byte("李强"))
	c2 := sha256.Sum256([]byte("关凤瑜"))
	fmt.Printf("%x\n%x\n%t\n%T\n", c1, c2, c1 == c2, c1)
	fmt.Printf("Diff: %d\n", diffBitsSha256(c1, c2))
	// Output:
	// 2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881
	// 4b68ab3847feda7d6c62c1fbcbeebfa35eab7351ed5e78f4ddadea5df64b8015
	// false
	// [32]uint8
	// Diff: 127
}

func diffBitsSha256(c1, c2 [32]uint8) int {
	res := 0
	for i := 0; i < 32; i++ {
		res += int(pc[c1[i]^c2[i]])
	}
	return res
}
```

sha256,sha384,sha512输出

```go
package main

import (
	"crypto/sha256"
	"crypto/sha512"
	"flag"
	"fmt"
	"os"
)

func main() {
	var sha int
	flag.IntVar(&sha, "sha", 256, "SHA")
	var str string
	str = os.Args[len(os.Args)-1]
	fmt.Println(str)
	flag.Parse()
	switch sha {
	case 256:
		OutputSHA256(str)
	case 384:
		OutputSHA384(str)
	case 512:
		OutputSHA512(str)
	}
}

func OutputSHA256(str string) {
	fmt.Printf("%x\n", sha256.Sum256([]byte(str)))
}
func OutputSHA384(str string) {
	fmt.Printf("%x\n", sha512.Sum384([]byte(str)))
}
func OutputSHA512(str string) {
	fmt.Printf("%x\n", sha512.Sum512([]byte(str)))
}
```

### Slice

语法

- 一个slice类型一般写作[]T，slice的语法和数组很像，只是没有固定长度而已。
- 初始化时，同样可以使用顺序指定序列或者通过索引指定。
- 不能用`==`、`!=`比较2个slice中的数据是否相同。
- 使用高度优化的bytes.Equal函数来判断两个字节型slice是否相等（[]byte），但是对于其他类型的slice，我们必须自己展开每个元素进行比较
- make
	```go
	make([]T, len)
	make([]T, len, cap) // same as make([]T, cap)[:len]
	```

底层数据结构

- slice是轻量级的数据结构，slice的底层引用了一个数组对象。
- slice由三个部分构成：指针、长度和容量。
- 长度对应slice中元素的数目；长度不能超过容量
- 容量一般是从slice的开始位置到底层数据的结尾位置
- 内置的len和cap函数分别返回slice的长度和容量。
- 切片操作超出cap(s)的上限将导致一个panic异常，但是超出len(s)则是意味着扩展了slice，因为新slice的长度会变大

底层实现

- 创建slice时，会隐式地创建一个合适大小的数组，然后slice的指针指向底层的数组。
- 多个slice之间可以共享底层的数据，并且引用的数组部分区间可能重叠。

零值

- 一个零值的slice等于nil。一个nil值的slice并没有底层数组。一个nil值的slice的长度和容量都是0
- 非nil值的slice的长度和容量也是0的，例如[]int{}或make([]int, 3)[3:]

Append

```go
var x []int
x = append(x, 1)
x = append(x, 2, 3)
x = append(x, 4, 5, 6)
x = append(x, x...) // append the slice x
fmt.Println(x)      // "[1 2 3 4 5 6 1 2 3 4 5 6]"
```
- 内存分配策略类似于C++的Vector

重写reverse函数，使用数组指针代替slice。

```go
package main

import "fmt"

// reverse reverses a slice of ints in place.
func reverseSlice(s []int) {
	for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
		s[i], s[j] = s[j], s[i]
	}
}

func reversePoint(p *[6]int) {
	for i, j := 0, len(*p)-1; i < j; i, j = i+1, j-1 {
		(*p)[i], (*p)[j] = (*p)[j], (*p)[i]
	}
}

func main() {
	a := [...]int{0, 1, 2, 3, 4, 5}
	reverseSlice(a[:])
	fmt.Println(a) // "[5 4 3 2 1 0]"
	reversePoint(&a)
	fmt.Println(a) // "[5 4 3 2 1 0]"
}
```

编写一个rotate函数，通过一次循环完成旋转。

```go
package main

import "fmt"

// reverse reverses a slice of ints in place.
func gcd(a, b int) int {
	for b > 0 {
		a, b = b, a%b
	}
	return a
}

// n > 0   <-------
// n < 0   ------->
func rotate(s []int, n int) {
	length := len(s)
	n = (n + length) % length
	gcd := gcd(length, n)
	loop := length / gcd

	for i := 0; i < gcd; i++ {
		temp := s[i]
		j := 0
		for ; j < loop-1; j++ {
			s[(i+j*n)%length] = s[(i+j*n+n)%length]
		}
		s[(i+j*n)%length] = temp
	}
}

func main() {
	a := [...]int{0, 1, 2, 3, 4, 5}
	rotate(a[:], 2)
	fmt.Println(a) // "[5 4 3 2 1 0]"
}
```

写一个函数在原地完成消除[]string中相邻重复的字符串的操作。

```go
package main

import "fmt"

func removeAdjRepeat(s []string) []string {
	for i := 0; i < len(s)-1; i++ {
		if s[i] == s[i+1] {
			copy(s[i:], s[i+1:])
			s = s[:len(s)-1]
			i--
		}
	}
	return s
}

func main() {
	s := []string{"1", "1", "2"}
	s = removeAdjRepeat(s)
	fmt.Println(s)
}
```

### Map

底层数据结构

- map是一个哈希表的引用
- map中所有的key都有相同的类型，所有的value也有着相同的类型
- map中的元素并不是一个变量，因此我们不能对map的元素进行取址操作：禁止对map元素取址的原因是map可能随着元素数量的增长而重新分配更大的内存空间，从而可能导致之前的地址无效。

遍历

- `for k, v := range map`
- 迭代顺序是不确定的

按顺序遍历key/value对方法

```go
import "sort"

keys := make([]string, 0, len(map))
for key := range map {
    keys = append(keys, key)
}
sort.Strings(keys)
for _, key := range keys {
    fmt.Printf("%s\t%d\n", key, map[key])
}
```

查找

```go
if age, ok := ages["bob"]; !ok { /* ... */ }
```

判断2个map是否相等

```go
func equal(x, y map[string]int) bool {
    if len(x) != len(y) {
        return false
    }
    for k, xv := range x {
        if yv, ok := y[k]; !ok || yv != xv {
            return false
        }
    }
    return true
}
```

### 结构体

语法

- 结构体成员名字是以大写字母开头的，那么该成员就是导出的；
- 一个命名为S的结构体类型将不能再包含S类型的成员：因为一个聚合的值不能包含它自身。但是S类型的结构体可以包含*S指针类型的成员，这可以让我们创建递归的数据结构，比如链表和树结构等。

比较

- 如果结构体的全部成员都是可以比较的，那么结构体也是可以用`==`比较的

结构体嵌入和匿名成员

- Go语言有一个特性让我们只声明一个成员对应的数据类型而不指名成员的名字；这类成员就叫匿名成员。
- 匿名成员的数据类型必须是命名的类型或指向一个命名的类型的指针。
```go
type Point struct {
    X, Y int
}

type Circle struct {
	Point
	Radius int
}

type Wheel struct {
	Circle
	Spokes int
}

var w Wheel
w.X = 8            // equivalent to w.Circle.Point.X = 8
w.Y = 8            // equivalent to w.Circle.Point.Y = 8
w.Radius = 5       // equivalent to w.Circle.Radius = 5
w.Spokes = 20

w = Wheel{Circle{Point{8, 8}, 5}, 20}

w = Wheel{
    Circle: Circle{
        Point:  Point{X: 8, Y: 8},
        Radius: 5,
    },
    Spokes: 20, // NOTE: trailing comma necessary here (and at Radius)
}
```
- 这样就可以直接访问叶子属性而不需要给出完整的路径。同时完整的访问方式同样支持`w.Circle.Point.X`
- 匿名成员也有一个隐式的名字，因此不能同时包含两个类型相同的匿名成员，这会导致名字冲突。

### JSON

marshaling:结构体slice转为JSON

```go
package main

import (
	"encoding/json"
	"fmt"
	"log"
)

type Movie struct {
	Title  string
	Year   int  `json:"released"`
	Color  bool `json:"color,omitempty"`
	Actors []string
}

var movies = []Movie{
	{Title: "Casablanca", Year: 1942, Color: false,
		Actors: []string{"Humphrey Bogart", "Ingrid Bergman"}},
	{Title: "Cool Hand Luke", Year: 1967, Color: true,
		Actors: []string{"Paul Newman"}},
	{Title: "Bullitt", Year: 1968, Color: true,
		Actors: []string{"Steve McQueen", "Jacqueline Bisset"}},
	// ...
}

func main() {
	data, err := json.Marshal(movies)
	if err != nil {
		log.Fatalf("JSON marshaling failed: %s", err)
	}
	fmt.Printf("%s\n", data)

	data, err = json.MarshalIndent(movies, "", "    ")
	if err != nil {
		log.Fatalf("JSON marshaling failed: %s", err)
	}
	fmt.Printf("%s\n", data)
}
```

- 成员Tag一般用原生字符串面值的形式书写
- json开头键名对应的值用于控制encoding/json包的编码和解码的行为，并且encoding/...下面其它的包也遵循这个约定
- 成员Tag中json对应值的第一部分用于指定JSON对象的名字
- omitempty选项，表示当Go语言结构体成员为空或零值时不生成该JSON对象（这里false为零值）

### 文本和HTML模板

```go
const templ = `{{.TotalCount}} issues:
{{range .Items}}----------------------------------------
Number: {{.Number}}
User:   {{.User.Login}}
Title:  {{.Title | printf "%.64s"}}
Age:    {{.CreatedAt | daysAgo}} days
{{end}}`
```

- 一个模板是一个字符串或一个文件，里面包含了一个或多个由双花括号包含的{{action}}对象。
- 每一个action，都有一个当前值的概念，对应点操作符，写作“.”。
- 模板中{{range .Items}}和{{end}}对应一个循环action，因此它们直接的内容可能会被展开多次，循环每次迭代的当前值对应当前的Items元素的值。
- 一个action中，|操作符表示将前一个表达式的结果作为后一个函数的输入，类似于UNIX中管道的概念。
- 对于Age部分，第二个动作是一个叫daysAgo的函数

## 函数

### 函数声明

Go语言没有默认参数值

### 递归

Go语言使用可变栈，栈的大小按需增加(初始时很小)。这使得我们使用递归时不必考虑溢出和安全问题。

### 多返回值

如果一个函数所有的返回值都有显式的变量名，那么该函数的return语句可以省略操作数。这称之为bare return。

```go
func HourMinSec(t time.Time) (hour, minute, second int)
```

### 错误

错误形式只有一种,ok

```go
value, ok := cache.Lookup(key)
if !ok {
    // ...cache[key] does not exist…
}
```

错误形式多种,error

```go
fmt.Println(err)
fmt.Printf("%v", err)
```

错误传递

```go
doc, err := html.Parse(resp.Body)
resp.Body.Close()
if err != nil {
    return nil, fmt.Errorf("parsing %s as HTML: %v", url,err)
}
```

- fmt.Errorf函数使用fmt.Sprintf格式化错误信息并返回
- 我们使用该函数添加额外的前缀上下文信息到原始错误信息。当错误最终由main函数处理时，错误信息应提供清晰的从原因到后果的因果链
- 错误信息中应避免大写和换行符。最终的错误信息可能很长

错误等待代码

```go
func WaitForServer(url string) error {
    const timeout = 1 * time.Minute
    deadline := time.Now().Add(timeout)
    for tries := 0; time.Now().Before(deadline); tries++ {
        _, err := http.Head(url)
        if err == nil {
            return nil // success
        }
        log.Printf("server not responding (%s);retrying…", err)
        time.Sleep(time.Second << uint(tries)) // exponential back-off
    }
    return fmt.Errorf("server %s failed to respond after %s", url, timeout)
}
```

错误打印

- log包中的所有函数会为没有换行符的字符串增加换行符

通过`io.EOF`错误判断文件结束

```go
in := bufio.NewReader(os.Stdin)
for {
    r, _, err := in.ReadRune()
    if err == io.EOF {
        break // finished reading
    }
    if err != nil {
        return fmt.Errorf("read failed:%v", err)
    }
    // ...use r…
}
```

### 函数值

- 函数被看作第一类值（first-class values）
- 函数类型的零值是nil。调用值为nil的函数值会引起panic错误
- 函数值可以与nil比较,但是函数值之间是不可比较的，也不能用函数值作为map的key

### 匿名函数

- 使用时注意循环变量的作用域问题

### 可变参数

```go
func sum(vals...int) int {
    total := 0
    for _, val := range vals {
        total += val
    }
    return total
}
fmt.Println(sum())           // "0"
fmt.Println(sum(3))          // "3"
fmt.Println(sum(1, 2, 3, 4)) // "10"
values := []int{1, 2, 3, 4}
fmt.Println(sum(values...)) // "10"
```

### Deferred函数

defer语句经常被用于处理成对的操作，如打开、关闭、连接、断开连接、加锁、释放锁。通过defer机制，不论函数逻辑多复杂，都能保证在任何执行路径下，资源被释放。释放资源的defer应该直接跟在请求资源的语句后。

### Panic函数

```
goroutine 1 [running]:
main.printStack()
src/gopl.io/ch5/defer2/defer.go:20
main.f(0)
src/gopl.io/ch5/defer2/defer.go:27
main.f(1)
src/gopl.io/ch5/defer2/defer.go:29
main.f(2)
src/gopl.io/ch5/defer2/defer.go:29
main.f(3)
src/gopl.io/ch5/defer2/defer.go:29
main.main()
src/gopl.io/ch5/defer2/defer.go:15
```

### Recover捕获异常

如果在deferred函数中调用了内置函数recover，并且定义该defer语句的函数发生了panic异常，recover会使程序从panic中恢复，并返回panic value。导致panic异常的函数不会继续运行，但能正常返回。在未发生panic时调用recover，recover会返回nil。

```go
func Parse(input string) (s *Syntax, err error) {
    defer func() {
        if p := recover(); p != nil {
            err = fmt.Errorf("internal error: %v", p)
        }
    }()
    // ...parser...
}
```

## 方法

### 方法声明

```go
package geometry

import "math"

type Point struct{ X, Y float64 }

// traditional function
func Distance(p, q Point) float64 {
    return math.Hypot(q.X-p.X, q.Y-p.Y)
}

// same thing, but as a method of the Point type
func (p Point) Distance(q Point) float64 {
    return math.Hypot(q.X-p.X, q.Y-p.Y)
}

// A Path is a journey connecting the points with straight lines.
type Path []Point
// Distance returns the distance traveled along the path.
func (path Path) Distance() float64 {
    sum := 0.0
    for i := range path {
        if i > 0 {
            sum += path[i-1].Distance(path[i])
        }
    }
    return sum
}
```

- Path是一个命名的slice类型，而不是Point那样的struct类型，然而我们依然可以为它定义方法。
- Go语言里，我们为一些简单的数值、字符串、slice、map来定义一些附加行为很方便。
- 不可以为指针和interface定义额外的方法

### 基于指针对象的方法

接收器`func (p *Point) ScaleBy(factor float64) {}`

### 通过潜入结构体来扩展类型

### 方法值和方法表达式

```go
type Rocket struct { /* ... */ }
func (r *Rocket) Launch() { /* ... */ }
r := new(Rocket)
time.AfterFunc(10 * time.Second, func() { r.Launch() }) //等效于下一行的方法值
time.AfterFunc(10 * time.Second, r.Launch)
```

```go
p := Point{1, 2}
q := Point{4, 6}

distance := Point.Distance   // method expression
fmt.Println(distance(p, q))  // "5"
fmt.Printf("%T\n", distance) // "func(Point, Point) float64"

scale := (*Point).ScaleBy
scale(&p, 2)
fmt.Println(p)            // "{2 4}"
fmt.Printf("%T\n", scale) // "func(*Point, float64)"

// 译注：这个Distance实际上是指定了Point对象为接收器的一个方法func (p Point) Distance()，
// 但通过Point.Distance得到的函数需要比实际的Distance方法多一个参数，
// 即其需要用第一个额外参数指定接收器，后面排列Distance方法的参数。
// 看起来本书中函数和方法的区别是指有没有接收器，而不像其他语言那样是指有没有返回值。
```

另一个方法表达式的例子

```go
type Point struct{ X, Y float64 }

func (p Point) Add(q Point) Point { return Point{p.X + q.X, p.Y + q.Y} }
func (p Point) Sub(q Point) Point { return Point{p.X - q.X, p.Y - q.Y} }

type Path []Point

func (path Path) TranslateBy(offset Point, add bool) {
    var op func(p, q Point) Point
    if add {
        op = Point.Add
    } else {
        op = Point.Sub
    }
    for i := range path {
        // Call either path[i].Add(offset) or path[i].Sub(offset).
        path[i] = op(path[i], offset)
    }
}
```

### 示例：Bit数组

### 封装

封装提供了三方面的优点。首先，因为调用方不能直接修改对象的变量值，其只需要关注少量的语句并且只要弄懂少量变量的可能的值即可。

第二，隐藏实现的细节，可以防止调用方依赖那些可能变化的具体实现，这样使设计包的程序员在不破坏对外的api情况下能得到更大的自由。

封装的第三个优点也是最重要的优点，是阻止了外部调用方对对象内部的值任意地进行修改。

## 接口

### 接口是合约

io.Writer

```go
package io

// Writer is the interface that wraps the basic Write method.
type Writer interface {
    // Write writes len(p) bytes from p to the underlying data stream.
    // It returns the number of bytes written from p (0 <= n <= len(p))
    // and any error encountered that caused the write to stop early.
    // Write must return a non-nil error if it returns n < len(p).
    // Write must not modify the slice data, even temporarily.
    //
    // Implementations must not retain p.
    Write(p []byte) (n int, err error)
}
```

- 这种类型都包含一个Write函数
- Fprintf接受任何满足io.Writer接口的值都可以工作

### 接口类型

```go
package io
type Reader interface {
    Read(p []byte) (n int, err error)
}
type Closer interface {
    Close() error
}
```

### 实现接口的条件

一个类型如果拥有一个接口需要的所有方法，那么这个类型就实现了这个接口。

io.ReadWriter或者io.Reader,io.Writer都是接口类型，更多方法的接口类型表示对实现它的类型要求更加严格。

interface{}被称为空接口类型。空接口类型对实现它的类型没有要求，所以我们可以将任意一个值赋给空接口类型。

interface{}没有任何方法，我们不能对它持有的值做操作。

### flag.Value接口

自定义flag.Value接口类型

```go
package main

import (
	"flag"
	"fmt"
)

type Celsius float64    // 摄氏温度
type Fahrenheit float64 // 华氏温度

func CToF(c Celsius) Fahrenheit { return Fahrenheit(c*9/5 + 32) }

func FToC(f Fahrenheit) Celsius { return Celsius((f - 32) * 5 / 9) }

func (c Celsius) String() string { return fmt.Sprintf("%g°C", c) }

type celsiusFlag struct{ Celsius }

func (f *celsiusFlag) Set(s string) error {
	var unit string
	var value float64
	fmt.Sscanf(s, "%f%s", &value, &unit) // no error check needed
	switch unit {
	case "C", "°C":
		f.Celsius = Celsius(value)
		return nil
	case "F", "°F":
		f.Celsius = FToC(Fahrenheit(value))
		return nil
	}
	return fmt.Errorf("invalid temperature %q", s)
}

func CelsiusFlag(name string, value Celsius, usage string) *Celsius {
	f := celsiusFlag{value}
	flag.CommandLine.Var(&f, name, usage)
	return &f.Celsius
}

var temp = CelsiusFlag("temp", 20.0, "the temperature")

func main() {
	flag.Parse()
	fmt.Println(*temp)
}
```

flag.Value接口类型定义如下

```go
package flag

// Value is the interface to the value stored in a flag.
type Value interface {
    String() string
    Set(string) error
}
```

- Set方法解析它的字符串参数并且更新标记变量的值
- celsiusFlag内嵌了一个Celsius类型，因此不用实现本身就已经有String方法了。为了实现flag.Value，我们只需要定义Set方法：
- CelsiusFlag函数将所有逻辑都封装在一起。它返回一个内嵌在celsiusFlag变量f中的Celsius指针给调用者
- 解释为什么帮助信息在它的默认值是20.0没有包含°C的情况下输出了°C。
	- 因为调用了String()方法

### 接口值

空类型

![img](http://upload-images.jianshu.io/upload_images/5952841-fe3dc30b28c0b78b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```go
var w io.Writer = os.Stdout
```

![img](http://upload-images.jianshu.io/upload_images/5952841-e3897806d97daf53.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```go
var x interface{} = time.Now()
```

![img](http://upload-images.jianshu.io/upload_images/5952841-608ec2da617643c6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### sort.Interface接口

```go
package main

import (
	"fmt"
	"os"
	"sort"
	"text/tabwriter"
	"time"
)

type Track struct {
	Title  string
	Artist string
	Album  string
	Year   int
	Length time.Duration
}

var tracks = []*Track{
	{"Go", "Delilah", "From the Roots Up", 2012, length("3m38s")},
	{"Go", "Moby", "Moby", 1992, length("3m37s")},
	{"Go Ahead", "Alicia Keys", "As I Am", 2007, length("4m36s")},
	{"Ready 2 Go", "Martin Solveig", "Smash", 2011, length("4m24s")},
}

func length(s string) time.Duration {
	d, err := time.ParseDuration(s)
	if err != nil {
		panic(s)
	}
	return d
}
func printTracks(tracks []*Track) {
	const format = "%v\t%v\t%v\t%v\t%v\t\n"
	tw := new(tabwriter.Writer).Init(os.Stdout, 0, 8, 2, ' ', 0)
	fmt.Fprintf(tw, format, "Title", "Artist", "Album", "Year", "Length")
	fmt.Fprintf(tw, format, "-----", "------", "-----", "----", "------")
	for _, t := range tracks {
		fmt.Fprintf(tw, format, t.Title, t.Artist, t.Album, t.Year, t.Length)
	}
	tw.Flush() // calculate column widths and print table
}

type byArtist []*Track

func (x byArtist) Len() int           { return len(x) }
func (x byArtist) Less(i, j int) bool { return x[i].Artist < x[j].Artist }
func (x byArtist) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }

func main() {
	sort.Sort(byArtist(tracks))
	printTracks(tracks)
	sort.Sort(sort.Reverse(byArtist(tracks)))
	printTracks(tracks)
}
```

### http.Handler接口

```go
package main

import (
	"fmt"
	"log"
	"net/http"
)

func main() {
	db := database{"shoes": 50, "socks": 5}
	mux := http.NewServeMux()
	mux.Handle("/list", http.HandlerFunc(db.list))
	mux.Handle("/price", http.HandlerFunc(db.price))
	log.Fatal(http.ListenAndServe("localhost:8000", mux))
}

type dollars float32

func (d dollars) String() string { return fmt.Sprintf("$%.2f", d) }

type database map[string]dollars

func (db database) list(w http.ResponseWriter, req *http.Request) {
	for item, price := range db {
		fmt.Fprintf(w, "%s: %s\n", item, price)
	}
}

func (db database) price(w http.ResponseWriter, req *http.Request) {
	item := req.URL.Query().Get("item")
	price, ok := db[item]
	if !ok {
		w.WriteHeader(http.StatusNotFound) // 404
		fmt.Fprintf(w, "no such item: %q\n", item)
		return
	}
	fmt.Fprintf(w, "%s\n", price)
}
```

### error接口

errors.New

fmt.Errorf

### 示例：表达式求值

### 类型断言

x.(T)被称为断言类型

- 如果断言的类型T是一个具体类型，然后类型断言检查x的动态类型是否和T相同。如果这个检查成功了，类型断言的结果是x的动态值，当然它的类型是T。如果检查失败，接下来这个操作会抛出panic。
- 如果相反地断言的类型T是一个接口类型，然后类型断言检查是否x的动态类型满足T。从一个接口类型转换到另一个接口类型，检查是否满足第二个接口类型的要求。
- 如果断言操作的对象是一个nil接口值，那么不论被断言的类型是什么这个类型断言都会失败

### 基于类型断言区别错误类型

```go
import (
    "errors"
    "syscall"
)

var ErrNotExist = errors.New("file does not exist")

// IsNotExist returns a boolean indicating whether the error is known to
// report that a file or directory does not exist. It is satisfied by
// ErrNotExist as well as some syscall errors.
func IsNotExist(err error) bool {
    if pe, ok := err.(*PathError); ok {
        err = pe.Err
    }
    return err == syscall.ENOENT || err == ErrNotExist
}
```

### 通过类型断言询问行为

### 类型分支

```go
switch x.(type) {
    case nil:       // ...
    case int, uint: // ...
    case bool:      // ...
    case string:    // ...
    default:        // ...
}
```

### 示例: 基于标记的XML解码

## Goroutines和Channels

顺序通信进程”(communicating sequential processes)或被简称为CSP

### Goroutines

当一个程序启动时，其主函数即在一个单独的goroutine中运行，我们叫它main goroutine。

除了从主函数退出或者直接终止程序之外，没有其它的编程方法能够让一个goroutine来打断另一个的执行。

等待转圈代码

```go
func spinner(delay time.Duration) {
    for {
        for _, r := range `-\|/` {
            fmt.Printf("\r%c", r)
            time.Sleep(delay)
        }
    }
}
```

### 示例: 并发的Clock服务

TCP服务器程序

```go
package main

import (
	"io"
	"log"
	"net"
	"time"
)

func main() {
	listener, err := net.Listen("tcp", "localhost:8000")
	if err != nil {
		log.Fatal(err)
	}

	for {
		conn, err := listener.Accept()
		if err != nil {
			log.Print(err) // e.g., connection aborted
			continue
		}
		go handleConn(conn) // handle one connection at a time
	}
}

func handleConn(c net.Conn) {
	defer c.Close()
	for {
		_, err := io.WriteString(c, time.Now().Format("15:04:05\n"))
		if err != nil {
			return // e.g., client disconnected
		}
		time.Sleep(1 * time.Second)
	}
}
```

TCP客户端程序

```go
package main

import (
	"io"
	"log"
	"net"
	"os"
)

func main() {
	conn, err := net.Dial("tcp", "localhost:8000")
	if err != nil {
		log.Fatal(err)
	}
	defer conn.Close()
	mustCopy(os.Stdout, conn)
}

func mustCopy(dst io.Writer, src io.Reader) {
	if _, err := io.Copy(dst, src); err != nil {
		log.Fatal(err)
	}
}
```

### 示例: 并发的Echo服务

### Channels

- channels则是goroutines之间的通信机制
- 可以让一个goroutine通过它给另一个goroutine发送值信息
- 每个channel都有一个特殊的类型，也就是channels可发送数据的类型
- 创建
```go
ch := make(chan int) // ch has type 'chan int'
```
- channel也对应一个make创建的底层数据结构的引用
- 两个相同类型的channel可以使用==运算符比较
- 一个发送语句将一个值从一个goroutine通过channel发送到另一个执行接收操作的goroutine
- 发送和接收两个操作都使用`<-`运算符
```go
ch <- x  // a send statement
x = <-ch // a receive expression in an assignment statement
<-ch     // a receive statement; result is discarded
```
- 关闭channel，随后对基于该channel的任何发送操作都将导致panic异常
- 试图重复关闭一个channel将导致panic异常
- 试图关闭一个nil值的channel也将导致panic异常
- 关闭一个channels还会触发一个广播机制
```go
close(ch)
```
- 对一个已经被close过的channel进行接收操作依然可以接受到之前已经成功发送的数据；如果channel中已经没有数据的话将直接返回并产生一个零值的数据。
- 创建带缓存的Channel
```go
ch = make(chan int)    // unbuffered channel
ch = make(chan int, 0) // unbuffered channel
ch = make(chan int, 3) // buffered channel with capacity 3
```

不带缓存的Channels

- 一个基于无缓存Channels的发送操作将导致发送者goroutine阻塞，直到另一个goroutine在相同的Channels上执行接收操作，当发送的值通过Channels成功传输之后，两个goroutine可以继续执行后面的语句。反之，如果接收操作先发生，那么接收者goroutine也将阻塞，直到有另一个goroutine在相同的Channels上执行发送操作。
- 基于无缓存Channels的发送和接收操作将导致两个goroutine做一次同步操作。因为这个原因，无缓存Channels有时候也被称为同步Channels。
- 当通过一个无缓存Channels发送数据时，接收者收到数据发生在唤醒发送者goroutine之前（译注：**happens before**，这是Go语言并发内存模型的一个关键术语！）。

串联的Channels（Pipeline）

- 一个Channel的输出作为下一个Channel的输入。这种串联的Channels就是所谓的管道（pipeline）。
- Channel可以被range

单方向的Channel

- 类型`chan<- int`表示一个只发送int的channel
- 类型`<-chan int`表示一个只接收int的channel
- 对一个只接收的channel调用close将是一个编译错误
- 任何双向channel向单向channel变量的赋值操作都将导致该隐式转换
- 不能将一个类似chan<- int类型的单向型的channel转换为chan int类型的双向型的channel

带缓存的Channels

- 向缓存Channel的发送操作就是向内部缓存队列的尾部插入元素，接收操作则是从队列的头部删除元素。如果内部缓存队列是满的，那么发送操作将阻塞直到因另一个goroutine执行接收操作而释放了新的队列空间。相反，如果channel是空的，接收操作将阻塞直到有另一个goroutine执行发送操作而向队列插入元素。
- 内置的cap函数获取缓存的容量`cap(ch)`
- 内置的len函数获取缓存内有效元素的个数`len(ch)`

goroutines泄漏，这是一个BUG。和垃圾变量不同，泄漏的goroutines并不会被自动回收，

### 并发的循环

注意循环变量失效问题

sync.WaitGroup

```go
// makeThumbnails6 makes thumbnails for each file received from the channel.
// It returns the number of bytes occupied by the files it creates.
func makeThumbnails6(filenames <-chan string) int64 {
    sizes := make(chan int64)
    var wg sync.WaitGroup // number of working goroutines
    for f := range filenames {
        wg.Add(1)
        // worker
        go func(f string) {
            defer wg.Done()
            thumb, err := thumbnail.ImageFile(f)
            if err != nil {
                log.Println(err)
                return
            }
            info, _ := os.Stat(thumb) // OK to ignore error
            sizes <- info.Size()
        }(f)
    }

    // closer
    go func() {
        wg.Wait()
        close(sizes)
    }()

    var total int64
    for size := range sizes {
        total += size
    }
    return total
}
```

- Add和Done方法的不对称。Add是为计数器加一，必须在worker goroutine开始之前调用，而不是在goroutine中
- Done却没有任何参数；其实它和Add(-1)是等价的。

### 并发的Web爬虫

### 基于select的多路复用

select{}，会永远地等待下去

多个case同时就绪时，select会随机地选择一个执行

### 示例: 并发的目录遍历

### 并发的退出

关闭一个channel来进行广播

```go
var done = make(chan struct{})

func cancelled() bool {
    select {
    case <-done:
        return true
    default:
        return false
    }
}
```

### 示例: 聊天服务

```go
package main

import (
	"bufio"
	"fmt"
	"log"
	"net"
)

func main() {
	listener, err := net.Listen("tcp", "localhost:8000")
	if err != nil {
		log.Fatal(err)
	}
	go broadcaster()
	for {
		conn, err := listener.Accept()
		if err != nil {
			log.Print(err)
			continue
		}
		go handleConn(conn)
	}
}

type client chan<- string // an outgoing message channel

var (
	entering = make(chan client)
	leaving  = make(chan client)
	messages = make(chan string) // all incoming client messages
)

func broadcaster() {
	clients := make(map[client]bool) // all connected clients
	for {
		select {
		case msg := <-messages:
			// Broadcast incoming message to all
			// clients' outgoing message channels.
			for cli := range clients {
				cli <- msg
			}
		case cli := <-entering:
			clients[cli] = true

		case cli := <-leaving:
			delete(clients, cli)
			close(cli)
		}
	}
}

func handleConn(conn net.Conn) {
	ch := make(chan string) // outgoing client messages
	go clientWriter(conn, ch)

	who := conn.RemoteAddr().String()
	ch <- "You are " + who
	messages <- who + " has arrived"
	entering <- ch

	input := bufio.NewScanner(conn)
	for input.Scan() {
		messages <- who + ": " + input.Text()
	}
	// NOTE: ignoring potential errors from input.Err()

	leaving <- ch
	messages <- who + " has left"
	conn.Close()
}

func clientWriter(conn net.Conn, ch <-chan string) {
	for msg := range ch {
		fmt.Fprintln(conn, msg) // NOTE: ignoring network errors
	}
}
```
