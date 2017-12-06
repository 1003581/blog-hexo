---
title: 图像处理岗位面试题搜罗汇总
date: 2017-09-14 15:59:00
tags: 
- 面试
- 机器视觉
categories: 机器学习
---

图像处理岗位面试题搜罗汇总
<!-- more -->

## Matlab编程

### Matlab 中读、写及显示一幅图像的命令各是什么？

imread(), imwrite(), imshow()

### Matlab 与VC++混合编程有哪几种方式？

Matlab引擎方式(Matlab后台程序为服务器，VC前端为客户端，C/S结构)、Matlab编译器（将Matlab源代码编译为C++可以调用的库文件）及COM组件（Matlab生成COM组件，VC调用）

### Matlab运算中 `.*`和 `*` 的区别？

`.*`表示矩阵元素分别相乘，要求两个矩阵具有相同的shape。`*`表示矩阵相乘。

## 图像处理基础部分

### Intel指令集中MMX,SSE,SSE2,SSE3和SSE4指的是什么？

MMX（Multi Media eXtension，多媒体扩展指令集）是一些整数并行运算指令。

SSE（Streaming SIMD Extensions，单指令多数据流扩展）是一系列浮点并行运算指令。

### 并行计算有哪些实现方式？

单指令多数据流SIMD、对称多处理机SMP、大规模并行处理机MPP、工作站机群COW、分布共享存储DSM多处理机。

### 彩色图像、灰度图像、二值图像和索引图像区别？

彩色图像：RGB图像。灰度图像：0-255像素值。二值图像：0和1，用于掩膜图像。

索引图像：在灰度图像中，自定义调色板，自定义输出256种颜色值。

### 常用边缘检测有哪些算子，各有什么特性？

1. **Sobel算子**：典型的基于一阶导数的边缘检测算子，对于像素的位置的影响做了加权，可以降低边缘模糊程度。
    
    不足：没有将图像的主体与背景严格地区分开来, 没有基于图像灰度进行处理.

    卷积核和像素更新公式如下：

    $$
    G_x=\left[ \begin{matrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{matrix} \right]
    G_y=\left[ \begin{matrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{matrix} \right]
    G = \sqrt{ {G_x}^2 + {G_y}^2 }
    $$
1. **Isotropic Sobel**：各向同性Sobel(Isotropic Sobel)算子。各向同性Sobel算子和普通Sobel算子相比，它的位置加权系数更为准确，在检测不同方向的边沿时梯度的幅度一致。

    卷积核和像素更新公式如下：

    $$
    G_x=\left[ \begin{matrix} -1 & 0 & +1 \\ -\sqrt2 & 0 & +\sqrt2 \\ -1 & 0 & +1 \end{matrix} \right]
    G_y=\left[ \begin{matrix} -1 & -\sqrt2 & -1 \\ 0 & 0 & 0 \\ +1 & +\sqrt2 & +1 \end{matrix} \right]
    G = \sqrt{ {G_x}^2 + {G_y}^2 }
    $$
1. **Roberts算子**：一种利用局部差分算子寻找边缘的算子。

    不足：对噪声敏感,无法抑制噪声的影响。

    卷积核和像素更新公式如下：

    $$
    G_x=\left[ \begin{matrix} -1 \\ +1 \end{matrix} \right]
    G_y=\left[ \begin{matrix} -1 & +1 \end{matrix} \right]
    G = \sqrt{ {G_x}^2 + {G_y}^2 }
    $$

    检测对角线方向梯度时：

    $$
    G_x=\left[ \begin{matrix} -1 & 0 \\ 0 & +1 \end{matrix} \right]
    G_y=\left[ \begin{matrix} 0 & -1 \\ +1 & 0 \end{matrix} \right]
    G = \sqrt{ {G_x}^2 + {G_y}^2 }
    $$
1. **Prewitt算子**：Sobel是该算子的改进版。

    卷积核和像素更新公式如下：

    $$
    G_x=\left[ \begin{matrix} -1 & 0 & +1 \\ -1 & 0 & +1 \\ -1 & 0 & +1 \end{matrix} \right]
    G_y=\left[ \begin{matrix} -1 & -1 & -1 \\ 0 & 0 & 0 \\ +1 & +1 & +1 \end{matrix} \right]
    G_x'=\left[ \begin{matrix} -1 & -1 & 0 \\ -1 & 0 & +1 \\ 0 & +1 & +1 \end{matrix} \right]
    G_y'=\left[ \begin{matrix} 0 & 1 & 1 \\ -1 & 0 & 1 \\ -1 & -1 & 0 \end{matrix} \right]
    $$
1. **Laplacian算子**：拉普拉斯算子,各向同性算子，二阶微分算子,只适用于无噪声图象,存在噪声情况下，使用Laplacian算子检测边缘之前需要先进行低通滤波。

    卷积核和像素更新公式如下：
    
    $$
    R=\left[ \begin{matrix} -1 & -1 & -1 \\ -1 & 8 & -1 \\ -1 & -1 & -1 \end{matrix} \right]
    G=\left\{ \begin{matrix} 1 & |R(x,y)| \ge T \\ 0 & others \end{matrix} \right.
    $$
1. **Canny算子**：一个具有滤波，增强，检测的多阶段的优化算子。先利用高斯平滑滤波器来平滑图像以除去噪声，采用一阶偏导的有限差分来计算梯度幅值和方向，然后再进行非极大值抑制。
1. **Laplacian of Gaussian(LoG)算子**：先对图像做高斯滤波，再做Laplacian算子检测。

### 简述BP神经网络

BP(back propagation)神经网络，输入X，通过隐藏节点的非线性变换后，输出信号Y，通过误差分析，来调整隐藏节点的W和b。

### AdBoost的基本原理？

AdBoost是一个广泛使用的BOOSTING算法，其中训练集上依次训练弱分类器，每次下一个弱分类器是在训练样本的不同权重集合上训练。权重是由每个样本分类的难度确定的。分类的难度是通过分类器的输出估计的。

## C/C++部分

### 关键字static的作用是什么？

1. 在函数体，一个被声明为静态的变量在这一函数被调用过程中维持其值不变。
1. 在模块内（但在函数体外），一个被声明为静态的变量可以被模块内所用函数访问，但不能被模块外其它函数，它是一个本地的全局变量。
1. 在模块内，一个被声明为静态的函数只可被这一模块的它函数调用。那就是，这个函数被限制在声明它的模块的本地范围内使用。

### 嵌入式系统总是用户对变量或寄存器进行位操作。给定一个整型变量a,写两段代码，第一个设置a的bit3，第二消除a的 bit 3。在以上两个操作中，要保持其它位不变.

```c++
#include <iostream>
#include <bitset>
using namespace std;

#define BIT3 (0x1<<3)
void set_bit3(unsigned &a)
{
	a |= BIT3;
}
void clear_bits(unsigned &a)
{
	a &= ~BIT3;
}

int main()
{
	unsigned a = UINT_MAX;
	clear_bits(a);
	cout << (bitset<32>)a << endl;
	set_bit3(a);
	cout << (bitset<32>)a << endl;
	return 0;
}
```

### 简述C，C++程序编译的内存分配情况？

C，C++中内存分配方式可以分为三种：  

1. 从静态存储区域分配：内存在程序编译时就已经分配好，这块内存在程序的整个运行期间都存在。速度快，不容易出错，因有系统自行管理。
1. 在栈上分配：在执行函数时，函数内局部变量的存储单元都在栈上创建，函数执行结束时这些存储单元自动被释放。栈内存分配运算内置于处理器的指令集中，效率很高，但是分配的内存容量有限。
1. 从堆上分配：即运态内存分配。程序在运行时候用malloc或new申请任意大小的内存，程序员自己负责在何进用free 和delete释放内存。

一个C、C++程序编译时内存分为5大存储区：堆区、栈区、全局区、文字常量区和程序代码区。

### 行列递增矩阵的查找

解法一、分治法

因为矩阵的行和列都是递增的，所以整个矩阵的对角线上的数字也是递增的，故我们可以在对角线上进行二分查找，如果要找的数是6介于对角线上相邻的两个数4、10，可以排除掉左上和右下的两个矩形，而在左下和右上的两个矩形继续递归查找

解法二、定位法

首先直接定位到最右上角的元素，比要找的数大就往左走，比要找数的小就往下走，直到找到要找的数字为止，走不动，说明这个数不存在。这个方法的时间复杂度O（m+n）。代码如下：

```c++
#include <iostream>
#include <vector>
using namespace std;

bool YoungMatrix(vector< vector<int> > mat, int target){
	int y = 0, x = mat[y].size() - 1;
	int var = mat[y][x];
	while (true) {
		if (var == target){
			printf("Mat[%d][%d]=%d\n", y, x, target);
			return true;
		}
		else if (var < target && y < mat.size() - 1){
			var = mat[++y][x];
		}
		else if (var > target && x > 0){
			var = mat[y][--x];
		}
		else{
			return false;
		}
	}
}

int main(){
	vector<vector<int> > matrix(20);
	for (int i = 0; i < 20; i++){
		for (int j = 0; j < 20; j++) {
			matrix[i].push_back(i+j);
			cout << matrix[i][j] << " ";
		}
		cout << endl;
	}
	cout << YoungMatrix(matrix, 38) << endl;
	return 0;
}
```

### 从1到500的500个数，第一次删除奇数位上的所有数，第二次删除剩下来的奇数位，以此类推，最后剩下的唯一一位数是什么？

就是当1~n，2^i<n<2^(i+1)时候，这样删除剩下的是2^i。2^8<500<2^9，所以剩下的就是2^8=256。

### 给出了一个n*n的矩形，编程求从左上角到右下角的路径数（n > =2），限制只能向右或向下移动，不能回退。例如当n=2时，有6条路径。

从左上角到右下角总共要走2n步，其中横向要走n步，所以总共就是$C_{2n}^n$次。

### 给出一棵二叉树的前序和中序遍历，输出后续遍历的结果。

已知一棵二叉树前序遍历和中序遍历分别为ABDEGCFH和DBGEACHF，则该二叉树的后序遍历为多少？

```c++
#include <iostream>
#include <string>
using namespace std;

string Subsequent(string pre, string mid) {
	if (pre.size() != mid.size() || pre.empty()) return "";
	char root = pre[0];
	int rootIndex = mid.find(root);
	string leftPre = pre.substr(1, rootIndex);
	string leftMid = mid.substr(0, rootIndex);
	string rightPre = pre.substr(rootIndex + 1);
	string rightMid = mid.substr(rootIndex + 1);
	
	string res;
	res += Subsequent(leftPre, leftMid);
	res += Subsequent(rightPre, rightMid);
	res += root;
	return res;
}

int main(){
	string pre = "ABDEGCFH";
	string mid = "DBGEACHF";
	cout << Subsequent(pre, mid) << endl;
	return 0;
}
```

### 自定义实现字符串转为整数的算法，例如把“123456”转成整数123456.(输入中可能存在符号，和数字)

```c++
#include <iostream>
using namespace std;

int strToInt(const char* str)
{
	long long result = 0;
	if (str != NULL) {
		const char* digit = str;

		bool minus = false;

		if (*digit == '+')
			digit++;
		else if (*digit == '-') {
			digit++;
			minus = true;
		}

		while (*digit != '\0') {
			if (*digit >= '0' && *digit <= '9') {
				result = result * 10 + (*digit - '0');
				if (result > numeric_limits<int>::max()) {
					result = 0;
					break;
				}
				digit++;
			}
			else {
				result = 0;
				break;
			}
		}

		if (*digit == '\0') {
			if (minus)
				result = 0 - result;
		}
	}
	return static_cast<int>(result);
}

int main(){
	cout << strToInt("-164546") << endl;
	return 0;
}
```

### 字符串最长公共子序列

动态规划推导式

![](https://box.kancloud.cn/2016-06-07_575683a585d0b.jpg)

```c++
#include <iostream>
#include <vector>
#include <string>
using namespace std;

string lcs(string s1, string s2) {
	int len1 = s1.size();
	int len2 = s2.size();
	vector< vector<int> > mat;
	vector< vector<int> > direct; //0-up 1-leftup 2-left
	mat.resize(len1 + 1);
	direct.resize(len1 + 1);
	for (int i = 0; i < len1 + 1; i++) {
		mat[i].resize(len2 + 1);
		direct[i].resize(len2 + 1);
	}
	for (int i = 0; i < len1 + 1; i++) {
		mat[i][0] = 0;
		direct[i][0] = 0;
	}
	for (int j = 0; j < len2 + 1; j++) {
		mat[0][j] = 0;
		direct[0][j] = 0;
	}
		
	for (int i = 1; i <= len1; i++) {
		for (int j = 1; j <= len2; j++) {
			if (s1[i - 1] == s2[j - 1]) {
				mat[i][j] = mat[i - 1][j - 1] + 1;
				direct[i][j] = 1;
			}

			else {
				if (mat[i][j - 1] > mat[i - 1][j]) {
					mat[i][j] = mat[i][j - 1];
					direct[i][j] = 2;
				}
				else {
					mat[i][j] = mat[i - 1][j];
					direct[i][j] = 0;
				}
			}
		}
	}
	cout << "lcs:" << mat[len1][len2] << endl;
	string res;
	int i = len1, j = len2;
	while (i > 0 && j > 0) {
		if (direct[i][j] == 1) {
			res = s1[i - 1] + res;
			i--;
			j--;
		}
		else if (direct[i][j] == 0) {
			i--;
		}
		else {
			j--;
		}
	}
	return res;
}

int main() {
	string s1 = "ABCBDAB";
	string s2 = "BDCABA";
	cout << lcs(s1, s2) << endl;
	return 0;
}
```

### 字符串最长公共子串

与上文区别是不等时的处理方式，和最后是整个矩阵中寻找最大值。

```c++
#include <iostream>
#include <vector>
#include <string>
using namespace std;

string lcs(string s1, string s2) {
	int len1 = s1.size();
	int len2 = s2.size();
	vector< vector<int> > mat;
	mat.resize(len1 + 1);
	for (int i = 0; i < len1 + 1; i++) {
		mat[i].resize(len2 + 1);
	}
	for (int i = 0; i < len1 + 1; i++) {
		mat[i][0] = 0;
	}
	for (int j = 0; j < len2 + 1; j++) {
		mat[0][j] = 0;
	}
		
	for (int i = 1; i <= len1; i++) {
		for (int j = 1; j <= len2; j++) {
			if (s1[i - 1] == s2[j - 1]) {
				mat[i][j] = mat[i - 1][j - 1] + 1;
			}
			else {
				mat[i][j] = 0;
			}
		}
	}

	int maxMat = 0;
	string res;
	for (int i = 1; i <= len1; i++) {
		for (int j = 1; j <= len2; j++) {
			if (mat[i][j] > maxMat) {
				maxMat = mat[i][j];
				res = s1.substr(i - maxMat, maxMat);
			}
		}
	}
	
	return res;
}

int main() {
	string s1 = "ABCBDAB";
	string s2 = "BDCABA";
	cout << lcs(s1, s2) << endl;
	return 0;
}
```

### 请实现一个函数：最长顺子。输入很多个整数(1<=数值<=13)，返回其中可能组成的最长的一个顺子(顺子中数的个数代表顺的长度)； 其中数字1也可以代表14；

直方图

```c++
#include <iostream>
#include <vector>
#include <string>
using namespace std;

vector<int> LongestShunZi(vector<int> input) {
	// 统计直方图
	vector<int> hist;
	hist.resize(15);
	for (int i = 0; i < input.size(); i++)
		if (input[i] > 0 && input[i] < 15)
			hist[input[i]] ++;
	hist[14] = hist[1];
	//最大牌数
	int maxCount = 0;
	for (int i = 1; i < 15; i++)
		if (hist[i] > maxCount)
			maxCount = hist[i];
	//求结果
	int resLen = 0;
	int resCount = 0;
	int resEnd = 0;
	for (int i = 1; i <= maxCount; i++)
	{
		int len = 0;
		int longestLen = 0;
		int longestEnd = 1;
		for (int j = 1; j < 15; j++) {
			if (hist[j] >= i) {
				len++;
				if (len > longestLen) {
					longestLen = len;
					longestEnd = j;
				}
			}
			else {
				len = 0;
			}
		}
		if (longestLen == 14 && 2 * i > hist[1]) longestLen--;
		if (longestLen * i > resLen * resCount) {
			resLen = longestLen;
			resCount = i;
			resEnd = longestEnd;
		}
	}

	vector<int> res;
	for (int i = resEnd - resLen + 1; i <= resEnd; i++)
		for (int j = 0; j < resCount; j++)
			res.push_back(i);
	return res;
}

int main() {
	int arr[] = { 1, 5, 2, 3, 4, 4, 5, 9, 6, 7, 2, 3, 3, 4 };
	vector<int> v(arr, arr+sizeof(arr)/sizeof(int));
	vector<int> res = LongestShunZi(v);
	for (int i = 0; i < res.size(); i++) cout << res[i] << " ";
	cout << endl;
	return 0;
}
```

## 软件编程部分

### 给你一个模块要求，你要做出这个模块，那么你的做出该模块的思路和步骤是什么？

### 对一批编号为1-100，全部开关朝上(开)的亮灯进行如下操作

对一批编号为1-100，全部开关朝上(开)的亮灯进行如下操作：凡是编号为1的倍数的灯反方向拨一次开关；凡是编号为2的倍数的灯反方向又拨一次开关；编号为3的倍数的灯反方向又拨一次开关……凡是编号为100的倍数的灯反方向拨一次开关。编写程序，模拟此过程，最后打印出所熄灭灯的编号。

### 实现个函数 `unsigned int convect(char* pstr)`

实现个函数 `unsigned int convect(char* pstr)`。其中`pstr`是十六进制数的字符串。函数`convect`将`pstr`转换成数字返回（比如：字符串'1A'，将返回数值26.注意，`pstr[0]`是'1'）。`pstr`中只有数字字符0到9、A到F。不得借助其它的系统函数调用。

### 实现一个函数`unsigned int counter(char* pstr)`

实现一个函数`unsigned int counter(char* pstr)`。函数将打印出匹配的括号对。比如：字符串"a(bc(d)ef(12)g)"就存在3对匹配的括号对，分别是：
1. 位置4上的（与位置6上的）匹配。打印4 6即可。
1. 位置9上的（与位置12上的）匹配。打印9 12即可。
1. 位置1上的（与位置14上的）匹配。打印1 14即可。

## 图像处理部分

### 图像融合

已知两幅拼接好的图像，两幅图像在几何关系配准之后，但两图之间存在明显灰度差别跳变，请设计一个算法对图像进行处理，让两幅图之间的灰度看不出跳变，形成自然过渡。（可以不考虑两图之间的黑图部分）。

![](http://img.blog.csdn.net/20131102205608671)

### 特征点匹配

如下图所示，请以准确快速实现配准为目标，设计算法，让两图中对应的特征点（至少一部分特征点）配准（即精准地地找出对应点之间对应的坐标关系值）。

![](http://img.blog.csdn.net/20131102205402359)

