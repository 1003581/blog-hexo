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

在建立树的过程中，利用平衡因子记录左右子树的高度差。

[理解](http://www.cnblogs.com/PerkinsZhu/p/5824015.html)

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

1. 将字符串分为独立的几个单词,并分别进行翻转。
2. 将反转后的结果再次反转。
3. 考虑了多个空格的反转情况。

```c++
#include <iostream>
#include <string>
using namespace std;

//将每个单词进行翻转  
void reverse(string &s, int low, int high)
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
    getline(cin, a);
    for (int i = 0; i <= a.size(); i++)
    {
        if (a[i] == ' ' || i == a.size())
        {
            reverse(a, i - num, i - 1);
            num = 0;
        }
        else
        {
            num++;
        }
    }
    reverse(a, 0, a.size() - 1);
    cout << a << endl;
    return 0;
}
```

## 实现STL的auto函数

auto类似于占位符，auto类型的变量由编译器在编译过程中确定其变量类型，由变量的右值确认auto的类型。

auto 变量必须在定义时初始化，类似于const关键字。

[理解](http://www.cnblogs.com/QG-whz/p/4951177.html)

## 两个排序数组的归并

先依次比较两个数组，其值较小的就传入新的数组。当这次比较完之后可能有一个数组的长度很长，留下一些数组，然后在新数组的末尾插入即可。

代码：

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

1. 使用string的find函数，str.find(substr)。返回值为找到的索引号，-1表示不包含。
2. 自己实现，截取a中取b长度相等的字符串，判断是否与b相等。

自己实现的代码

```c++
bool isContain(char *a, char *b){  
    int i = 0;  
    int j = 0;  
    int k = 0;  
      
    while(a[i] != '\0' && b[j] != '\0'){  
        k = i+1;  
        while(b[j] != '\0'){  
            if(a[i] == b[j]){  
                i++;  
                j++;  
            }else{  
                i = k;  
                j = 0;  
                break;  
            }  
        }  
    }  
      
    return (j != 0);  
}  
```

## 实现阶乘

1. 递归写法。n超过一定程度时，int越界导致结果错误。
2. 用数组实现。用数组来弥补int超界问题，得手动处理进位问题。

递归写法

```c++
long fac(int n)
{  
    long f;  
    if(n<0) return -1;
    else if(n==0) return 1;  
    else if(n==1) return 1;  
    else f=n*fac(n-1);
    return f;  
}  
```

数组写法

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

使用动态规划

dp表代表当s1在i处是交错的，同时s2在j处是交错的时，s3在i+j处是否是交错的？

如果s1和s2在当前位置是空，s3也是空，则我们视为true;

如果s1是空，s2之前的位置是交错的而且s2在当前位置和s3的当前位置字符是一样的，则视为true;

反之s2为空时情况是一样的。

现在考虑s1和s2都不为空的情况。当我们从(i-1,j)到达(i,j)处时，如果(i-1,j)处是交错的而i处与当前的s3一致，则视为true;

当我们从(i,j-1)到达(i,j)处时，如果(i,j-1)处是交错的，且j处与当前的s3一致，则视为true;

```c++
#include <iostream>
#include <string>
using namespace std;

#define N 1000

bool IsOK(char *s1, char *s2, char *s3)
{
    int len1 = strlen(s1);
    int len2 = strlen(s2);
    bool dp[N][N];
    memset(dp, 0, N*N);
    for (int i = 0; i <= len1; ++i)
    {
        for (int j = 0; j <= len2; ++j)
        {
            if (i == 0 && j == 0)
                dp[i][j] = true;
            else if (i == 0)
                dp[i][j] = (dp[i][j - 1] && s2[j - 1] == s3[i + j - 1] );
            else if (j == 0)
                dp[i][j] = (dp[i - 1][j] && s1[i - 1] == s3[i + j - 1] );
            else
                dp[i][j] = (dp[i - 1][j] && s1[i - 1] == s3[i + j - 1] )
                        || (dp[i][j - 1] && s2[j - 1] == s3[i + j - 1] );
        }
    }
    return dp[len1][len2];
}

int main()
{
    char str1[] = "abce";
    char str2[] = "dfg";
    char str3[] = "abcdefg";
    cout << IsOK(str1, str2, str3) <<endl;
    return 0;
}
```

## 字符串反转

第一种：使用string.h中的strrev函数

```c++
#include <iostream>  
#include <cstring>  
using namespace std;  
  
int main()  
{  
    char s[]="hello";  
    strrev(s);  
    cout<<s<<endl;  
    return 0;  
} 
```

第二种：使用algorithm中的reverse函数

```c++
#include <iostream>
#include <string>
#include <algorithm>
using namespace std;

int main()
{
    string s = "hello";
    reverse(s.begin(),s.end());
    cout<<s<<endl;
    return 0;
}
```

第三种：自己编写

```c++
#include <iostream>  
using namespace std;  
  
void Reverse(char *s,int n)
{  
    for(int i=0,j=n-1; i<j; i++,j--)
    {  
        char c=s[i];  
        s[i]=s[j];  
        s[j]=c;
    }  
}  
  
int main()  
{  
    char s[]="hello";  
    Reverse(s,5);  
    cout<<s<<endl;  
    return 0;  
}
```

## 二叉树中序遍历

1、递归法

递归的终止条件是当前节点是否为空。首先递归调用遍历左子树，然后访问当前节点，最后递归调用右子树。

```
void PrintMed(TreeNode* root)
{  
    if(root == NULL) return;  
    PrintMed(ret,root->left);  
    cout<< root->val << " ";
    PrintMed(ret,root->right);
}
```

2、非递归

从根节点开始找二叉树的最左节点，将走过的节点保存在一个栈中，找到最左节点后开始访问。

对于每个节点来说，它都是以自己为根的子树的根节点，访问完自己之后就可以访问右子树了。

这种方法时间复杂度是O(n)，空间复杂度也是O(n)。

```
void PrintMed(TreeNode* root) 
{
    if (root == NULL) return;
    TreeNode *curr=root;  
    stack<TreeNode*> st;  
    while(!st.empty() || curr != NULL)  
    {  
        while(curr != NULL)  
        {  
            st.push(curr);  
            curr = curr->left;  
        }  
        curr = st.top();  
        st.pop();  
        cout << curr->val << " ";  
        curr = curr->right;  
    }  
    return ret;  
}  
```

3、Morris法

这种方法不使用递归，不使用栈，O(1)的空间复杂度，O(n)的时间复杂度完成二叉树的遍历。这种方法的基本思路就是将所有右儿子为NULL的节点的右儿子指向后继节点（对于右儿子不为空的节点，右儿子就是接下来要访问的节点）。这样，对于任意一个节点，当访问完它后，它的右儿子已经指向了下一个该访问的节点。对于最右节点，不需要进行这样的操作。注意，这样的操作是在遍历的时候完成的，完成访问节点后会把树还原。整个循环的判断条件为当前节点是否为空。例如上面的二叉树，遍历过程如下（根据当前节点c的位置）：

（1）当前节点为10，因为左儿子非空，不能访问，找到c的左子树的最右节点p：

（2）找节点c的左子树的最右节点有两种终止条件，一种右儿子为空，一种右儿子指向当前节点。下面是右儿子为空的情况，这种情况先要构造，将节点p的右儿子指向后继节点c，然后c下移：

（3）当前节点c的左儿子为空，进行访问。访问后将c指向右儿子（即后继节点）：

（4）继续寻找左子树的最右节点，这次的终止条件是最右节点为当前节点。这说明当前节点的左子树遍历完毕，访问当前节点后，还原二叉树，将当前节点指向后继节点：

（5）重复上述过程，直到c指向整棵二叉树的最右节点：

## 实现int atoi(char *)

```c++
#include <iostream>
#include <limits>
using namespace std;

int Atoi(char *str)
{
    if (str == NULL) return 0;

    //工作指针
    char *p = str;

    //去除字符串前的空格
    while (*p != 0 && *p == ' ') p++;

    //空字符串
    if (*p == 0) return 0;

    //正负号的考虑
    bool positive = true;
    if (*p == '+') p++;
    else if (*p == '-') 
    {
        p++;
        positive = false;
    }

    long long result = 0;

    while (*p != 0)
    {
        //如果当前字符不是数字，则返回
        if ((*p < '0') || (*p > '9'))
            return 0;

        //如果当前字符是数字则计算数值
        result = result * 10 + (*p - '0');

        //缓冲区溢出
        //正数不能大于INT_MAX，负数不能大于INT_MIN
        if ((positive&&result > INT_MAX) || (!positive&&result*(-1) < INT_MIN))
            return 0;

        //移到下一个字符
        p++;
    }

    return positive == true ? result : ((-1)*result);
}

int main()
{
    cout << Atoi("999999") << endl;
    return 0;
}
```

## 字符串中第一个出现一次的字符串

解题思路：以空间换时间

我们可以定义一个哈希表（外部空间），其键值（Key）是字符，而值（Value）是该字符出现的次数。

同时我们还需要从头开始扫描字符串两次：

1）第一次扫描字符串时，每扫描到一个字符就在哈希表的对应项中把次数加1。（时间效率O(n)）

2）第二次扫描时，每扫描到一个字符就能从哈希表中得到该字符出现的次数。这样第一个只出现一次的字符就是符合要求的输出。（时间效率O(n)）

```c++
#include <iostream>
#include <string>
using namespace std;

char FirstNotRepeatingChar(string str)
{
    if (str == "") return '\0';

    // 借助数组来模拟哈希表，只用1K的空间消耗
    int hash[256];
    memset(hash, 0, sizeof(int) * 256);

    for (int i = 0; i < str.size(); i++)
    {
        hash[str[i]]++;
    }

    for (int i = 0; i < str.size(); i++)
    {
        if (hash[str[i]] == 1)
        {
            return str[i];
        }
    }
    return '\0';
}

int main()
{
    cout << FirstNotRepeatingChar("abcdea") << endl;
    return 0;
}
```

## 一个排序数组，重复放到数组后，前面有序，后面可无序

## 连续重复最长子串 abcabcdef => abc，两个字符重复次数最多的子串

## 复制一个每个节点都有随机指针的单链表

第一遍遍历生成所有新节点时同时建立一个原节点和新节点的哈希表，第二遍给随机指针赋值时，查找时间是常数级。

```c++
class Solution {
public:
    RandomListNode *copyRandomList(RandomListNode *head) {
        if (!head) return NULL;
        RandomListNode *res = new RandomListNode(head->label);
        RandomListNode *node = res;
        RandomListNode *cur = head->next;
        map<RandomListNode*, RandomListNode*> m;
        m[head] = res;
        while (cur) {
            RandomListNode *tmp = new RandomListNode(cur->label);
            node->next = tmp;
            m[cur] = tmp;
            node = node->next;
            cur = cur->next;
        }
        node = res;
        cur = head;
        while (node) {
            node->random = m[cur->random];
            node = node->next;
            cur = cur->next;
        }
        return res;
    }
};
```

## 字符串空格换成"20%"，转换后内存不够怎么追加，若连续的空格看成一个空格，怎么处理

转换后内存不够，可以先计算转换后需要的大小，然后用realloc函数进行重新分配空间。

连续空格看成一个空格，只转换第一个空格，剩余空格放弃拷贝。

## O(1)删除一个单链表的结点

把下一个节点的数据拷贝到自己这个节点，然后删除下一个节点，可以得到同样的效果。

若遇到最后一个节点，还是需要遍历一遍单链表。

```c++
//在O(1)时间内，删除一个节点，函数如下  
void DeleteNodeNumone(ListNode** phead,ListNode* pToBeDelete)  
{  
    if (*phead == nullptr || pToBeDelete == nullptr)  
        return;  
  
    //删除非尾节点  
    if (pToBeDelete->_next != nullptr)  
    {  
        ListNode* temp = pToBeDelete->_next;  
        pToBeDelete->_data = temp->_data;  
        pToBeDelete->_next = temp->_next;  
  
        delete temp;  
        temp = nullptr;  
    }  
  
    //只有一个节点  
    else if (*phead == pToBeDelete)  
    {  
        delete pToBeDelete;  
        pToBeDelete = nullptr;  
        *phead = nullptr;  
    }  
  
    //最后一种，删除节点是尾节点  
    else  
    {  
        ListNode* cur = *phead;  
        while (cur->_next != pToBeDelete)  
        {  
            cur = cur->_next;  
        }  
        delete pToBeDelete;  
        pToBeDelete = nullptr;  
        cur->_next = nullptr;  
    }  
}  
```

## 两个队列模拟一个栈

实现一:

s1是入栈的，s2是出栈的。

入队列，直接压到s1是就行了

出队列，先把s1中的元素全部出栈压入到s2中，弹出s2中的栈顶元素；再把s2的所有元素全部压回s1中

实现二:

s1是入栈的，s2是出栈的。保证所有元素都在一个栈里面

入队列时：如果s1为空，把s2中所有的元素倒出压到s1中；否则直接压入s1

出队列时：如果s2不为空，把s2中的栈顶元素直接弹出；否则，把s1的所有元素全部弹出压入s2中，再弹出s2的栈顶元素

实现三

s1是入栈的，s2是出栈的。

入队列：直接压入s1即可

出队列：如果s2不为空，把s2中的栈顶元素直接弹出；否则，把s1的所有元素全部弹出压入s2中，再弹出s2的栈顶元素

## 数组，负在前，正在后

扫描这个数组的时候，如果发现有正数出现在负数的前面，则交换他们的顺序。因此我们可以维护两个指针，第一个指针初始化为数组的第一个数字，它只向后移动；第二个指针初始化为数组的最后一个数字，它只向前移动。在两个指针相遇之 前，第一个指针总是位于第二个指针的前面。如果第一个指针指向的数字是正而第二个指针指向的数字是负数，我们就交换这两个数字。时间复杂度为O(n),空间复杂度为O(1).

## 数组中出现次数超过一半的数

方法1：

使用map统计每个数出现的次数，输出出现次数大于数组长度一半的数字。

方法2：

借鉴快速排序算法，通过Partition()返回index，如果index==mid，那么就表明找到了数组的中位数；如果indexmid，表明中位数在[start,index-1]之间。知道最后求得index==mid循环结束。

## 字符串最小组成子字符串的重复个数

abcabc => 2

abcabcd => 1

## 不用额外空间交换2个数字

方法1：算术运算（加减）：

```c++
void swap1(int &a, int &b){
    a = a + b;
    b = a - b;
    a = a - b;
}
```

方法2：乘除法。

```c++
void swap2(int &a, int &b){
    a = a * b;
    b = a / b;
    a = a / b;
}
```

方法3：异或法。

```c++
void swap3(int &a, int &b){
    a ^= b;         //x先存x和y两者的信息
    b ^= a;         //保持x不变，利用x异或反转y的原始值使其等于x的原始值
    a ^= b;         //保持y不变，利用x异或反转y的原始值使其等于y的原始值
}
```