---
title: 一些面试题目
date: 2017-09-26 19:00:00
tags: 
- oj
categories: c++
---

一些面试题目

<!-- more -->

## 平衡二叉树

## 不用加减法交换2个整数

异或法

```c++
void swap3(int &a, int &b){
    a ^= b; //x先存x和y两者的信息
    b ^= a; //保持x不变，利用x异或反转y的原始值使其等于x的原始值
    a ^= b; //保持y不变，利用x异或反转y的原始值使其等于y的原始值
}
```

## hello world转为world hello

三步反转法

1. 将字符串分为独立的几个单词,并分别进行翻转
2. 将反转后的结果再次反转

```c++
#include <iostream>
#include <string>
using namespace std;  
  
//将每个单词进行翻转  
void reverse(string &s,int low,int high)  
{  
    while (low < high)  
    {  
        char tmp = s[high];  
        s[high] = s[low];
        s[low] = tmp;
        low++;
        high--;
    }
}  
  
int main()  
{  
    int num = 0;  
    string a;
    cin>>a;  
    cout<<a<<endl;
    for (int i = 0; i <= a.size(); i++)  
    {  
        if (a[i] == ' ' || i == a.size())  
        {  
            reverse(a, i-num, i-1);  
            num = 0;
        }  
        else  
        {  
            num++;  
        }  
    }
    reverse(a, 0, a.size()-1);
    cout<<a<<endl;  
    return 0;  
}  
```

## 实现STL的auto函数

## 两个排序数组的归并

```c++
#include <iostream>
#include <vector>
using namespace std;

//a,b数组必须有序
void Merge(vector<int> a, vector<int> b, vector<int> &c)
{
    c.resize(a.size()+b.size());

    int lenA = 0, lenB = 0, lenC = 0;

    while(lenA < a.size() && lenB < b.size())
    {
        if(a[lenA] < b[lenB])
        {
            c[lenC++] = a[lenA++];
        }
        else
        {
            c[lenC++] = b[lenB++];
        }
    }
    while(lenA < a.size())
    {
        c[lenC++] = a[lenA++];
    }

    while(lenB < b.size())
    {
        c[lenC++] = b[lenB++];
    }
}

int main() 
{
    int arrA[] = {1,2,3,3,4,5};
    vector<int> a(arrA, arrA + sizeof(arrA)/sizeof(int));
    int arrB[]={7,8,8,9};
    vector<int> b(arrB, arrB + sizeof(arrB)/sizeof(int));

    vector<int> c;
    Merge(a,b,c);
    for (int i=0; i<c.size(); i++) 
    {
        cout<<c[i]<<" ";
    }
    cout<<endl;
    return 0;
}
```

## 字符串包含

## 实现阶乘

```c++
#include <iostream>  
using namespace std;  
int main()  
{  
    int n, i, j;  
    while (cin >> n)  
    {  
        int flag = 1;  
        int carry = 0;        //设置进位  
        int res = 0;  
        int str[1000000];     //根据题目要求设置数组的大小  
        str[0] = 1;  
        for (i=2; i <= n; i++)   //从2开始计算阶乘  
        {  
            for (j=1; j <= flag; j++)     //根据进位flag大小来判断当前阶乘结果的的位数  
            {  
                res = str[j-1] * i + carry;  
                str[j-1] = res % 10;  
                carry = res/10;
            }  
            while (carry)  //当进位大于一时将结果扩展到下一位  
            {  
                flag++;  
                str[flag-1] = carry % 10;  
                carry/=10;  
            }  
        }  
        for (i=flag-1; i>=0; i--)    //将结果数组倒序输出，注意最后一位是"i=flag-1",因为"flag=1" 对应str[0],以此类推  
        {  
             cout<<str[i];
        }  
        cout<<endl;
    }  
    return 0;  
}
```

## string s1,s2,s3。判断s3能否由s1和s2组成

```c++
#include <iostream>
#include <string>
using namespace std;  

bool IsOK(char *str1, char *str2, char *str3)
{
    int len1 = strlen(str1);
    int len2 = strlen(str2);
    

    for(int i = 0 ; i <= len1 ; ++i)
    {  
        for(int j = 0 ; j <= len2 ; ++j)
        {  
            //当c[i+j-1] == a[i-1]时,需要看一下c[i+j-1]是否能由a[i-2]和b[j-1]组成  
            if(i >= 1 && str3[i+j-1] == str1[i-1] && can[i-1][j] == 1)
            {
                can[i][j] = 1;  
            }
            else if(j >= 1 && str3[i+j-1] == str2[j-1] &&  can[i][j-1] == 1)
            {  
                can[i][j] = 1;  
            }  
        }  
    }
    return can[len1][len2];
}

int main()
{  
    char str1[] = "abce";
    char str2[] = "dfg";
    char str3[] = "abcdefg";
    cout<<IsOK(str1,str2,str3);
    return 0;  

} 
```


## 字符串反转

## 二叉树中序遍历

## 实现int atoi(char *)

## 字符串中第一个出现一次的字符串

## 一个排序数组，重复放到数组后，前面有序，后面可无序

## 连续重复最长子串 abcabcdef => abc，两个字符重复次数最多的子串

## 复制一个每个节点都有随机指针的单链表

## 字符串空格换成"20%"，转换后内存不够怎么追加，若连续的空格看成一个空格，怎么处理

## O(1)删除一个单链表的结点

## 两个队列模拟一个栈

## 数组，负在前，正在后

## 数组中出现次数超过一半的数

## 字符串最小组成子字符串的重复个数

abcabc => 2

abcabcd => 1

## 不用额外空间交换2个数字