---
title: Lua语言学习
date: 2017-09-14 16:01:45
tags: lua
categories: openresty
---

## Lua

<!-- more -->

### 网上教程

[Lua简明教程](http://coolshell.cn/articles/10739.html)

[极客学院Openresty教程](http://wiki.jikexueyuan.com/project/openresty/)

[Windows安装](http://wiki.jikexueyuan.com/project/openresty/openresty/install.html)

使用 `tasklist /fi "imagename eq nginx.exe"` 命令查看 nginx 进程，其中一个是 master 进程，另一个是 worker 进程

cmd进入nginx的安装目录，输入`start nginx.exe`来启动nginx

cmd进入nginx的安装目录，输入`nginx -s reload`来重启nginx

### LuaJIT部分语法

#### 环境搭建

```shell
wget http://luajit.org/download/LuaJIT-2.1.0-beta1.tar.gz
tar -xvf LuaJIT-2.1.0-beta1.tar.gz
cd LuaJIT-2.1.0-beta1
make
sudo make install

luajit -v
```

#### 基础数据类型

nil

无效值

boolean

`false`和`nil`为假，其余包括0全部为真。

number

实数类型，可以用`math.floor`向下取整和`math.cell`向上取整。

string

`'abc'`，`"abc"`，`[=[abc\n]=]`(不转义)

字符串不可改变，只能新建，相同字符串在内存中只有一个副本

table

```lua
local corp = {
    web = "www.google.com",   --索引为字符串，key = "web",
                              --            value = "www.google.com"
    telephone = "12345678",   --索引为字符串
    staff = {"Jack", "Scott", "Gary"}, --索引为字符串，值也是一个表
    100876,              --相当于 [1] = 100876，此时索引为数字
                         --      key = 1, value = 100876
    100191,              --相当于 [2] = 100191，此时索引为数字
    [10] = 360,          --直接把数字索引给出
    ["city"] = "Beijing" --索引为字符串
}

print(corp.web)               -->output:www.google.com
print(corp["telephone"])      -->output:12345678
print(corp[2])                -->output:100191
print(corp["city"])           -->output:"Beijing"
print(corp.staff[1])          -->output:Jack
print(corp[10])               -->output:360
```

function

```lua
function foo()
end

foo = function ()
end
```

#### 表达式

`a and b` 如果`a`为nil，则返回a，否则返回b
`a or b` 如果`a`为nil，则返回b，否则返回a
`not a` 返回true or false

字符串拼接

```lua
print("Hello " .. "World")    -->打印 Hello World
print(0 .. 1)                 -->打印 01 自动将数字转化为string

str1 = string.format("%s-%s","hello","world")
print(str1)              -->打印 hello-world

str2 = string.format("%d-%s-%.2f",123,"world",1.21)
print(str2)              -->打印 123-world-1.21
```

字符串拼接会不断的创建新字符串，若在循环中进行拼接，则对性能有较大影响，推荐使用`table.concat()`

```lua
local pieces = {}
for i, elem in ipairs(my_list) do
    pieces[i] = my_process(elem)
end
local res = table.concat(pieces)
```

#### 控制语句

if

```lua
x = 10
if x > 0 then
    print("x is a positive number")
end
```

if-else

```lua
x = 10
if x > 0 then
    print("x is a positive number")
else
    print("x is a non-positive number")
end
```

if-elseif-else

```lua
score = 90
if score == 100 then
    print("Very good!Your score is 100")
elseif score >= 60 then
    print("Congratulations, you have passed it,your score greater or equal to 60")
--此处可以添加多个elseif
else
    print("Sorry, you do not pass the exam! ")
end
```

嵌套if

```lua
score = 0
if score == 100 then
    print("Very good!Your score is 100")
elseif score >= 60 then
    print("Congratulations, you have passed it,your score greater or equal to 60")
else
    if score > 0 then
        print("Your score is better than 0")
    else
        print("My God, your score turned out to be 0")
    end --与上一示例代码不同的是，此处要添加一个end
end
```

while

有`break`，无`continue`

```lua
while 表达式 do
--body
end
```

repeat

一直执行，直到`条件`为真

```lua
repeat
    body
until 条件
```

for

数字 for（numeric for）

```lua
for var = begin, finish, step do
    --body
end

for i = 1, 5 do
  print(i)
end

-- output:
1
2
3
4
5
```

1. var从begin取到finish，左右都为闭区间
1. begin、finish、step 三个表达式只会在循环开始时执行一次
1. 第三个表达式 step 是可选的，默认为 1
1. 控制变量 var 的作用域仅在 for 循环内，需要在外面控制，则需将值赋给一个新的变量

范型 for（generic for）

泛型 for 循环通过一个迭代器（iterator）函数来遍历所有值：

```lua
-- 打印数组a的所有值
local a = {"a", "b", "c", "d"}
for i, v in ipairs(a) do
  print("index:", i, " value:", v)
end

-- output:
index:  1  value: a
index:  2  value: b
index:  3  value: c
index:  4  value: d

-- 打印table t中所有的key
for k in pairs(t) do
    print(k)
end
```

#### 函数

```lua
function function_name (arc)  -- arc 表示参数列表，函数的参数列表可以为空
   -- body
end
```

Lua函数的参数大部分是按**值**传递的，只有传入一个**表table**时，会进行**引用**传递。

参数补齐

```lua
local function fun1(a, b)       --两个形参，多余的实参被忽略掉
   print(a, b)
end

local function fun2(a, b, c, d) --四个形参，没有被实参初始化的形参，用nil初始化
   print(a, b, c, d)
end

local x = 1
local y = 2
local z = 3

fun1(x, y, z)         -- z被函数fun1忽略掉了，参数变成 x, y
fun2(x, y, z)         -- 后面自动加上一个nil，参数变成 x, y, z, nil

-->output
1   2
1   2   3   ni
```

变长参数

```lua
local function func( ... )                -- 形参为 ... ,表示函数采用变长参数

   local temp = {...}                     -- 访问的时候也要使用 ...
   local ans = table.concat(temp, " ")    -- 使用 table.concat 库函数对数
                                          -- 组内容使用 " " 拼接成字符串。
   print(ans)
end

func(1, 2)        -- 传递了两个参数
func(1, 2, 3, 4)  -- 传递了四个参数

-->output
1 2

1 2 3 4
```

返回值

```lua
local function init()       -- init 函数 返回两个值 1 和 "lua"
    return 1, "lua"
end

local x, y, z = init(), 2   -- init 函数的位置不在最后，此时只返回 1
print(x, y, z)              -->output  1  2  nil

local a, b, c = 2, init()   -- init 函数的位置在最后，此时返回 1 和 "lua"
print(a, b, c)              -->output  2  1  lua

-- 使用括号运算符

print((init()), 2)   -->output  1  2
print(2, (init()))   -->output  2  1
```

全动态函数调用

```lua
local function run(x, y)
    print('run', x, y)
end

local function attack(targetId)
    print('targetId', targetId)
end

local function do_action(method, ...)
    local args = {...} or {}
    method(unpack(args, 1, table.maxn(args)))
end

do_action(run, 1, 2)         -- output: run 1 2
do_action(attack, 1111)      -- output: targetId    1111
```

#### 模块

my.lua

```lua
local foo={}

local function getname()
    return "Lucy"
end

function foo.greeting()
    print("hello " .. getname())
end

return foo
```

main.lua

```lua
local fp = require("my")
fp.greeting()     -->output: hello Lucy
```

#### String库

string.byte

返回ASCII码

用该函数来进行字符串相关的扫描和分析是最为高效的

```lua
print(string.byte("abc", 1, 3))
print(string.byte("abc", 3)) -- 缺少第三个参数，第三个参数默认与第二个相同，此时为 3
print(string.byte("abc"))    -- 缺少第二个和第三个参数，此时这两个参数都默认为 1

-->output
97  98  99
99
97
```

string.char (...)

接收 0 个或更多的整数（整数范围：0~255），返回这些整数所对应的 ASCII 码字符组成的字符串。当参数为空时，默认是一个 0。

```lua
print(string.char(96, 97, 98))
print(string.char())        -- 参数为空，默认是一个0，
                            -- 你可以用string.byte(string.char())测试一下
print(string.char(65, 66))

--> output
`ab

AB
```

string.upper(s)

接收一个字符串 s，返回一个把所有小写字母变成大写字母的字符串。

```lua
print(string.upper("Hello Lua"))  -->output  HELLO LUA
```

string.lower(s)

接收一个字符串 s，返回一个把所有大写字母变成小写字母的字符串。

```lua
print(string.lower("Hello Lua"))  -->output   hello lua
```

string.len(s)

接收一个字符串，返回它的长度。不推荐使用，推荐使用`#s`的方式获取字符串长度

```lua
print(string.len("hello lua")) -->output  9
```

string.find(s, p [, init [, plain]])

在 s 字符串中第一次匹配 p 字符串。若匹配成功，则返回 p 字符串在 s 字符串中出现的开始位置和结束位置；若匹配失败，则返回 nil。 第三个参数 init 默认为 1，并且可以为负整数，当 init 为负数时，表示从 s 字符串的 string.len(s) + init 索引处开始向后匹配字符串 p 。 第四个参数默认为 false，当其为 true 时，只会把 p 看成一个字符串对待。

```lua
local find = string.find
print(find("abc cba", "ab"))
print(find("abc cba", "ab", 2))     -- 从索引为2的位置开始匹配字符串：ab
print(find("abc cba", "ba", -1))    -- 从索引为7的位置开始匹配字符串：ba
print(find("abc cba", "ba", -3))    -- 从索引为6的位置开始匹配字符串：ba
print(find("abc cba", "(%a+)", 1))  -- 从索引为1处匹配最长连续且只含字母的字符串
print(find("abc cba", "(%a+)", 1, true)) --从索引为1的位置开始匹配字符串：(%a+)

-->output
1   2
nil
nil
6   7
1   3   abc
nil
```

string.format(formatstring, ...)

按照格式化参数 formatstring，返回后面 ... 内容的格式化版本。编写格式化字符串的规则与标准 c 语言中 printf 函数的规则基本相同：它由常规文本和指示组成，这些指示控制了每个参数应放到格式化结果的什么位置，及如何放入它们。一个指示由字符 % 加上一个字母组成，这些字母指定了如何格式化参数，例如 d 用于十进制数、x 用于十六进制数、o 用于八进制数、f 用于浮点数、s 用于字符串等。在字符 % 和字母之间可以再指定一些其他选项，用于控制格式的细节。

```lua
print(string.format("%.4f", 3.1415926))     -- 保留4位小数
print(string.format("%d %x %o", 31, 31, 31))-- 十进制数31转换成不同进制
d = 29; m = 7; y = 2015                     -- 一行包含几个语句，用；分开
print(string.format("%s %02d/%02d/%d", "today is:", d, m, y))

-->output
3.1416
31 1f 37
today is: 29/07/2015
```

string.match(s, p , init)

在字符串 s 中匹配（模式）字符串 p，若匹配成功，则返回目标字符串中与模式匹配的子串；否则返回 nil。第三个参数 init 默认为 1，并且可以为负整数，当 init 为负数时，表示从 s 字符串的 string.len(s) + init 索引处开始向后匹配字符串 p。尽量用`ngx.re.match`替代。

```lua
print(string.match("hello lua", "lua"))
print(string.match("lua lua", "lua", 2))  --匹配后面那个lua
print(string.match("lua lua", "hello"))
print(string.match("today is 27/7/2015", "%d+/%d+/%d+"))

-->output
lua
lua
nil
27/7/2015
```

string.gmatch(s, p)

返回一个迭代器函数，通过这个迭代器函数可以遍历到在字符串 s 中出现模式串 p 的所有地方。尽量用`ngx.re.gmatch`替代。

```
s = "hello world from Lua"
for w in string.gmatch(s, "%a+") do  --匹配最长连续且只含字母的字符串
    print(w)
end

-->output
hello
world
from
Lua

t = {}
s = "from=world, to=Lua"
for k, v in string.gmatch(s, "(%a+)=(%a+)") do  --匹配两个最长连续且只含字母的
    t[k] = v                                    --字符串，它们之间用等号连接
end
for k, v in pairs(t) do
print (k,v)
end

-->output
to      Lua
from    world
```

string.rep(s, n)

返回字符串 s 的 n 次拷贝。

```lua
print(string.rep("abc", 3)) --拷贝3次"abc"

-->output  abcabcabc
```

string.sub(s, i [, j])

返回字符串 s 中，索引 i 到索引 j 之间的子字符串。当 j 缺省时，默认为 -1，也就是字符串 s 的最后位置。i 可以为负数。当索引 i 在字符串 s 的位置在索引 j 的后面时，将返回一个空字符串。

```lua
print(string.sub("Hello Lua", 4, 7))
print(string.sub("Hello Lua", 2))
print(string.sub("Hello Lua", 2, 1))    --看到返回什么了吗
print(string.sub("Hello Lua", -3, -1))

-->output
lo L
ello Lua

Lua
```

string.gsub(s, p, r [, n])

将目标字符串 s 中所有的子串 p 替换成字符串 r。可选参数 n，表示限制替换次数。返回值有两个，第一个是被替换后的字符串，第二个是替换了多少次。用`ngx.re.gsub`替换。

```lua
print(string.gsub("Lua Lua Lua", "Lua", "hello"))
print(string.gsub("Lua Lua Lua", "Lua", "hello", 2)) --指明第四个参数

-->output
hello hello hello   3
hello hello Lua     2
```

string.reverse (s)

接收一个字符串 s，返回这个字符串的反转。

#### Table库

下标从 1 开始

不要在 Lua 的 table 中使用 nil 值，如果一个元素要删除，直接 remove，不要用 nil 去代替。

table.concat (table [, sep [, i [, j ] ] ])

对于元素是 string 或者 number 类型的表 table，返回 table[i]..sep..table[i+1] ··· sep..table[j] 连接成的字符串。填充字符串 sep 默认为空白字符串。起始索引位置 i 默认为 1，结束索引位置 j 默认是 table 的长度。如果 i 大于 j，返回一个空字符串。

```lua
local a = {1, 3, 5, "hello" }
print(table.concat(a))              -- output: 135hello
print(table.concat(a, "|"))         -- output: 1|3|5|hello
print(table.concat(a, " ", 4, 2))   -- output:
print(table.concat(a, " ", 2, 4))   -- output: 3 5 hello
```

table.insert (table, [pos ,] value)

在（数组型）表 table 的 pos 索引位置插入 value，其它元素向后移动到空的地方。pos 的默认值是表的长度加一，即默认是插在表的最后。

table.remove (table [, pos])

在表 table 中删除索引为 pos（pos 只能是 number 型）的元素，并返回这个被删除的元素，它后面所有元素的索引值都会减一。pos 的默认值是表的长度，即默认是删除表的最后一个元素。

table.sort (table [, comp])

按照给定的比较函数 comp 给表 table 排序，也就是从 table[1] 到 table[n]，这里 n 表示 table 的长度。 比较函数有两个参数，如果希望第一个参数排在第二个的前面，就应该返回 true，否则返回 false。 如果比较函数 comp 没有给出，默认从小到大排序。

```lua
local function compare(x, y) --从大到小排序
   return x > y         --如果第一个参数大于第二个就返回true，否则返回false
end

local a = { 1, 7, 3, 4, 25}
table.sort(a)           --默认从小到大排序
print(a[1], a[2], a[3], a[4], a[5])
table.sort(a, compare) --使用比较函数进行排序
print(a[1], a[2], a[3], a[4], a[5])

-->output
1   3   4   7   25
25  7   4   3   1
```

#### 其他

##### [日期时间函数](http://wiki.jikexueyuan.com/project/openresty/lua/time_date_function.html)

##### [数学库](http://wiki.jikexueyuan.com/project/openresty/lua/math_library.html)

##### [文件操作](http://wiki.jikexueyuan.com/project/openresty/lua/file.html)

##### [元素](http://wiki.jikexueyuan.com/project/openresty/lua/metatable.html)

##### [面向对象编程](http://wiki.jikexueyuan.com/project/openresty/lua/object_oriented.html)

##### [局部变量](http://wiki.jikexueyuan.com/project/openresty/lua/local.html)

##### [判断数组大小](http://wiki.jikexueyuan.com/project/openresty/lua/array_size.html)

##### [非空判断](http://wiki.jikexueyuan.com/project/openresty/lua/not_nil.html)

##### [正则表达式](http://wiki.jikexueyuan.com/project/openresty/lua/re.html)

##### [不用标准库](http://wiki.jikexueyuan.com/project/openresty/lua/not_use_lib.html)

##### [虚变量](http://wiki.jikexueyuan.com/project/openresty/lua/dummy_var.html)

##### [抵制使用module()定义模块](http://wiki.jikexueyuan.com/project/openresty/lua/not_use_module.html)

##### [调用代码前先定义函数](http://wiki.jikexueyuan.com/project/openresty/lua/function_before_use.html)

##### [点号与冒号操作符的区别](http://wiki.jikexueyuan.com/project/openresty/lua/dot_diff.html)

##### [module是邪恶的](http://wiki.jikexueyuan.com/project/openresty/lua/module_is_evil.html)

##### [FFI](http://wiki.jikexueyuan.com/project/openresty/lua/FFI.html)

##### [什么是JIT](http://wiki.jikexueyuan.com/project/openresty/lua/what_jit.html)

