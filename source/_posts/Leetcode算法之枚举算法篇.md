---
title: Leetcode算法之枚举算法篇
date: 2024-09-16 00:00:00
updated: 2024-10-17 00:00:00
tags:
  - Datawhale
abbrlink: 9e17a883
---

# Datwhale|力扣刷题之枚举算法

> 

作者：@同济大学 刘越

Github ID：@miracle-techlink

联系邮箱：[miracle.techlink@gmail.com](mailto:miracle.techlink@gmail.com)

本文为Datawhale组队学习之Leetcode刷题笔记，感谢Datawhale提供的学习资源以及组队学习的小伙伴们的讨论与帮助。

本文教程链接如下：[Github地址](https://github.com/datawhalechina/leetcode-notes/blob/main/docs/ch04/index.md)

**PS:** 原教程使用python编写，而本文使用C++语言。

## 枚举算法简介

枚举算法的核心思想是：通过列举问题的所有状态，将它们逐一与目标状态进行比较，从而得到满足条件的解。

由于枚举算法要通过列举问题的所有状态来得到满足条件的解，因此，在问题规模变大时，其效率一般是比较低的。**但是枚举算法也有自己特有的优点：**

- 多数情况下容易编程实现，也容易调试。

- 建立在考察大量状态、甚至是穷举所有状态的基础上，所以算法的正确性比较容易证明。

所以，枚举算法通常用于求解问题规模比较小的问题，或者作为求解问题的一个子算法出现，通过枚举一些信息并进行保存，而这些消息的有无对主算法效率的高低有着较大影响。

## 枚举算法的解题思路

### 枚举算法解题步骤

**采用枚举算法解题的一般思路如下:**

> 

- 确定枚举对象、枚举范围和判断条件，并判断条件设立的正确性。

- 一一枚举可能的情况，并验证是否是问题的解。

- 考虑提高枚举算法的效率。

**注意事项：**

> 

- 枚举算法的时间复杂度通常很高，因为它可能需要检查解空间中的所有元素。

- 对于大规模问题，枚举算法可能不实际，因为它可能需要过多的计算资源。

- 可以通过剪枝技术（即在枚举过程中提前排除不可能的解）来优化枚举算法。

枚举算法适用于解空间较小的问题，对于解空间大的问题，通常需要结合其他算法技术来提高效率。

### 枚举算法的解题模板

下面举个著名的例子：「百钱买百鸡问题」。这个问题是我国古代数学家张丘在「算经」一书中提出的。该问题叙述如下：

> 

今有鸡翁一，值钱五；鸡母一，值钱三；鸡雏三，值钱一；百钱买百鸡，问鸡翁、母、雏各几何？

下面我们根据算法的一般思路来解决一下这道题。

**1.确定枚举对象、枚举范围和判断条件，并判断条件设立的正确性。**

- 

确定枚举对象：枚举对象为公鸡、母鸡、小鸡的只数，那么我们可以用变量 、、 分别来代表公鸡、母鸡、小鸡的只数。

- 

确定枚举范围：因为总共买了  元，所以公鸡、母鸡、小鸡的只数之和不超过 ，即 。又因为公鸡、母鸡、小鸡的只数必须为正整数，所以 、、 的枚举范围为 。

- 

确定判断条件：根据题意，我们可以列出两个方程式：，。在枚举 、、 的过程中，我们可以根据这两个方程式来判断是否当前状态是否满足题意。

**2.一一枚举可能的情况，并验证是否是问题的解。**

根据上面的分析，我们可以使用三重循环来枚举 、、 的所有可能情况，并在枚举过程中判断当前状态是否满足题意。如果满足题意，则输出当前状态。

```c++
#include <iostream>
using namespace std;

int main() {
    for (int x = 0; x <= 100; x++) {
        for (int y = 0; y <= 100; y++) {
            for (int z = 0; z <= 100; z++) {
                if (5 * x + 3 * y + z / 3 == 100 && x + y + z == 100) {
                    cout << "公鸡：" << x << "只，母鸡：" << y << "只，小鸡：" << z << "只" << endl;
                }
            }
        }
    }
    return 0;
}
```

**3.考虑提高枚举算法的效率。**

原来的取值范围都是，但是我们可以通过分析题目来缩小取值范围。比如，所有钱买公鸡，最多能买只，所以的取值范围是。同理，的取值范围是，的取值范围是。这样，我们就可以将枚举算法的效率提高一倍。

```c++
#include <iostream>
using namespace std;

int main() {
    for (int x = 0; x <= 20; x++) {
        for (int y = 0; y <= 33; y++) {
            int z = 100 - x - y;
            if (5 * x + 3 * y + z / 3 == 100) {
                cout << "公鸡：" << x << "只，母鸡：" << y << "只，小鸡：" << z << "只" << endl;
            }
        }
    }
    return 0;           
}
```

## 枚举算法LeetCode题目

**题目预览如下：**

[](https://leetcode.cn/problems/two-sum/)[](https://datawhalechina.github.io/leetcode-notes/#/solutions/0001)[](https://github.com/datawhalechina/leetcode-notes/blob/main/docs/solutions/0001.md)
[](https://leetcode.cn/problems/count-primes/)[](https://datawhalechina.github.io/leetcode-notes/#/solutions/0204)[](https://github.com/datawhalechina/leetcode-notes/blob/main/docs/solutions/0204.md)
[](https://leetcode.cn/problems/count-square-sum-triples/)[](https://datawhalechina.github.io/leetcode-notes/#/solutions/1925)[](https://github.com/datawhalechina/leetcode-notes/blob/main/docs/solutions/1925.md)
[](https://leetcode.cn/problems/number-of-students-doing-homework-at-a-given-time/)[](https://datawhalechina.github.io/leetcode-notes/#/solutions/1450)[](https://github.com/datawhalechina/leetcode-notes/blob/main/docs/solutions/1450.md)
[](https://leetcode.cn/problems/coordinate-with-maximum-network-quality/)[](https://leetcode.cn/problems/coordinate-with-maximum-network-quality/description/)[](https://github.com/datawhalechina/leetcode-notes/blob/main/docs/solutions/1620.md)
[](https://leetcode.cn/problems/number-of-common-factors/)[](https://datawhalechina.github.io/leetcode-notes/#/solutions/2427)[](https://github.com/datawhalechina/leetcode-notes/blob/main/docs/solutions/2427.md)
[](https://leetcode.cn/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)[](https://datawhalechina.github.io/leetcode-notes/#/solutions/Offer-57-II)[](https://github.com/datawhalechina/leetcode-notes/blob/main/docs/solutions/Offer-57-II.md)
[](https://leetcode.cn/problems/count-lattice-points-inside-a-circle/)[](https://datawhalechina.github.io/leetcode-notes/#/solutions/2249)[](https://github.com/datawhalechina/leetcode-notes/blob/main/docs/solutions/2249.md)
[](https://leetcode.cn/problems/similar-rgb-color/)[](https://datawhalechina.github.io/leetcode-notes/#/solutions/0800)[](https://github.com/datawhalechina/leetcode-notes/blob/main/docs/solutions/0800.md)
[](https://leetcode.cn/problems/maximal-square/)[](https://datawhalechina.github.io/leetcode-notes/#/solutions/0221)[](https://github.com/datawhalechina/leetcode-notes/blob/main/docs/solutions/0221.md)
[](https://leetcode.cn/problems/subarray-sum-equals-k/)[](https://datawhalechina.github.io/leetcode-notes/#/solutions/0560)[](https://github.com/datawhalechina/leetcode-notes/blob/main/docs/solutions/0560.md)

| 0560 | 和为 K 的子数组 | 网页链接 | 数组、哈希表、前缀和 | 中等 |
| --- | --- | --- | --- | --- |

### 0001 两数之和

**题目描述：** 给定一个整数数组 nums 和一个整数目标值 target。

**题目要求：** 在该数组中找出和为 target 的两个整数，并输出这两个整数的下标。可以按任意顺序返回答案

**说明：**

> 

- 

- 

- 

- 只会存在一个有效答案

**示例1：**

```plaintext
输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。
```

**示例2：**

```plaintext
输入：nums = [3,2,4], target = 6
输出：[1,2]
```

**思路1解析：** 暴力法，两层循环遍历数组，找到和为target的两个数。时间复杂度，空间复杂度。
**思路1代码：**

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        for (int i = 0; i < nums.size(); i++) {
            for (int j = i + 1; j < nums.size(); j++) {
                if (nums[i] + nums[j] == target) {
                    return {i, j};
                }
            }
        }
        return {};
    }
};
```

**思路2解析：** 哈希表，使用哈希表存储数组中的元素及其下标，然后遍历数组，找到和为target的两个数。时间复杂度，空间复杂度。
**思路2代码：**

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> hashMap;
        for (int i = 0; i < nums.size(); i++) {
            int complement = target - nums[i];
            if (hashMap.find(complement) != hashMap.end()) {
                return {hashMap[complement], i};
            }
            hashMap[nums[i]] = i;
        }
        return {};
    }
};
```

### 0204 计数质数

**题目描述：** 给定一个整数 n ，返回所有小于 n 的质数的数量。

**题目要求：** 质数是指大于1的自然数，且只有1和它本身两个因数。

**说明：**

> 

- 

**示例1：**

```plaintext
输入：n = 10
输出：4
```

**示例2：**

```plaintext
输入：n = 1
输出：0
```

**思路1解析：** 暴力法，两层循环遍历数组，找到和为target的两个数。时间复杂度，空间复杂度。
**思路1代码：**

```c++
class Solution {
public:
    int countPrimes(int n) {
        if (n <= 2) return 0;
        vector<bool> isPrime(n, true);
        int count = 0;
        for (int i = 2; i < n; i++) {
            if (isPrime[i]) {
                count++;
                for (int j = i * i; j < n; j += i) {
                    isPrime[j] = false;
                }
            }
        }
        return count;
    }
};
```

**思路2解析：** 埃氏筛，使用埃氏筛法找到所有小于n的质数。时间复杂度，空间复杂度。
**思路2代码：**     

```c++
class Solution {
public:
    int countPrimes(int n) {
        if (n <= 2) return 0;
        vector<bool> isPrime(n, true);
        int count = 0;
        for (int i = 2; i < n; i++) {
            if (isPrime[i]) {
                count++;
                for (int j = i * i; j < n; j += i) {
                    isPrime[j] = false;
                }
            }
        }
        return count;
    }
};
```

### 1925 统计平方和三元组的数目

**题目描述：** 给你一个整数数组 nums ，返回满足以下条件的三元组 (i, j, k) 的数量：

- 0 <= i < j < k < nums.length

- nums[i]2 + nums[j]2 = nums[k]2

**题目要求：** 返回满足条件的三元组数目。
**说明：**

> 

- 

- 

**示例1：**

```plaintext
输入：nums = [2,3,4,6,8,12]
输出：3
```

**示例2：**

```plaintext
输入：nums = [5,12,13,17,24]
输出：0
```

**思路1解析：** 暴力法，三层循环遍历数组，找到和为target的两个数。时间复杂度，空间复杂度。
**思路1代码：**

```c++
class Solution {
public:
    int countTriples(vector<int>& nums) {
        int count = 0;
        for (int i = 0; i < nums.size(); i++) {
            for (int j = i + 1; j < nums.size(); j++) {
                for (int k = j + 1; k < nums.size(); k++) { 
                    if (nums[i] * nums[i] + nums[j] * nums[j] == nums[k] * nums[k]) {
                        count++;
                    }
                }
            }
        }
        return count;
    }
};      
```

**思路2解析：** 枚举法，枚举所有可能的平方和，然后判断是否为平方数。时间复杂度，空间复杂度。
**思路2代码：**

```c++
class Solution {
public:
    int countTriples(vector<int>& nums) {   
        int count = 0;
        for (int i = 0; i < nums.size(); i++) {
            for (int j = i + 1; j < nums.size(); j++) {
                int sum = nums[i] * nums[i] + nums[j] * nums[j];
                int sqrtSum = sqrt(sum);
                if (sqrtSum * sqrtSum == sum) {
                    count++;
                }
            }                   
        }
        return count;
    }
};
```

### 1450 在既定时间做作业的学生人数

**题目描述：** 给你两个整数数组 startTime 和 endTime ，其中 startTime[i] 是第 i 个学生开始做作业的时间，endTime[i] 是第 i 个学生做完作业的时间。
**题目要求：** 返回在既定时间 queryTime 时，有多少个学生的作业既是在做还是在做。

**说明：**

> 

- 

- 

- 

**示例1：**

```plaintext
输入：startTime = [1,2,3], endTime = [3,2,7], queryTime = 4
输出：1
解释：
- 第一个学生从 startTime[0] = 1 开始做作业，并在 endTime[0] = 3 完成作业，在 queryTime = 4 时仍在做作业。
- 第二个学生从 startTime[1] = 2 开始做作业，并在 endTime[1] = 2 完成作业，在 queryTime = 4 时已完成作业。
- 第三个学生从 startTime[2] = 3 开始做作业，并在 endTime[2] = 7 完成作业，在 queryTime = 4 时仍在做作业。
因此，只有第一个学生在 queryTime 时仍在做作业。
```

**示例2：**

```plaintext
输入：startTime = [4], endTime = [4], queryTime = 4
输出：1
解释：
- 只有一个学生，他从 startTime[0] = 4 开始做作业，并在 endTime[0] = 4 完成作业，在 queryTime = 4 时已完成作业。
```

**思路1解析：** 暴力法，两层循环遍历数组，找到和为target的两个数。时间复杂度，空间复杂度。
**思路1代码：**

```c++
class Solution {
public:
    int busyStudent(vector<int>& startTime, vector<int>& endTime, int queryTime) {
        int count = 0;
        for (int i = 0; i < startTime.size(); i++) {
            if (startTime[i] <= queryTime && endTime[i] >= queryTime) {
                count++;
            }
        }
        return count;   
    }
};
```

### 2427 公因子的数目

**题目描述：** 给你两个正整数 a 和 b ，返回 a 和 b 的公因子的数目。
**题目要求：** 公因子是指能同时整除 a 和 b 的正整数。

**说明：**

> 

- 

**示例1：**

```plaintext
输入：a = 12, b = 6
输出：4
```

**示例2：**

```plaintext
输入：a = 25, b = 30
输出：2
解释：
- 12 和 6 的公因子有 1, 2, 3, 6。
- 25 和 30 的公因子有 1, 5。
```

**思路1解析：** 暴力法，两层循环遍历数组，找到和为target的两个数。时间复杂度，空间复杂度。

```c++
class Solution {
public:
    int commonFactors(int a, int b) {
        int count = 0;
        for (int i = 1; i <= min(a, b); i++) {
            if (a % i == 0 && b % i == 0) {
                count++;
            }
        }
        return count;
    }
};  
```

**思路2解析：线段树**
因为，所以我们可以维护一个区间为的线段树，初始化所有区间值都为。
然后遍历所有学生的开始时间和结束时间，并将区间值加。
在线段树中查询 对应的单点区间的最大值为多少。

时间复杂度：，其中  为数组元素的个数。每次更新和查询操作的时间复杂度都是 ，总共进行  次操作。
空间复杂度：，用于存储线段树。
**思路2代码：**

```c++
class SegTreeNode {
public:
    int left, right;
    int val;
    int lazy_tag;
    
    SegTreeNode() : left(-1), right(-1), val(0), lazy_tag(0) {}
};

class SegmentTree {
private:
    vector<SegTreeNode> tree;
    int size;
    
    void pushup(int index) {
        tree[index].val = max(tree[index * 2 + 1].val, tree[index * 2 + 2].val);
    }
    
    void pushdown(int index) {
        if (tree[index].lazy_tag == 0) return;
        
        int left_index = index * 2 + 1;
        int right_index = index * 2 + 2;
        
        tree[left_index].lazy_tag += tree[index].lazy_tag;
        tree[right_index].lazy_tag += tree[index].lazy_tag;
        
        tree[left_index].val += tree[index].lazy_tag;
        tree[right_index].val += tree[index].lazy_tag;
        
        tree[index].lazy_tag = 0;
    }
    
    void build(int index, int left, int right) {
        tree[index].left = left;
        tree[index].right = right;
        if (left == right) return;
        
        int mid = left + (right - left) / 2;
        build(index * 2 + 1, left, mid);
        build(index * 2 + 2, mid + 1, right);
    }
    
    void update_interval(int index, int q_left, int q_right, int val) {
        int left = tree[index].left;
        int right = tree[index].right;
        
        if (left >= q_left && right <= q_right) {
            tree[index].lazy_tag += val;
            tree[index].val += val;
            return;
        }
        
        pushdown(index);
        
        int mid = left + (right - left) / 2;
        if (q_left <= mid) {
            update_interval(index * 2 + 1, q_left, q_right, val);
        }
        if (q_right > mid) {
            update_interval(index * 2 + 2, q_left, q_right, val);
        }
        
        pushup(index);
    }
    
    int query_interval(int index, int q_left, int q_right) {
        int left = tree[index].left;
        int right = tree[index].right;
        
        if (left >= q_left && right <= q_right) {
            return tree[index].val;
        }
        
        pushdown(index);
        
        int mid = left + (right - left) / 2;
        int res = 0;
        if (q_left <= mid) {
            res = max(res, query_interval(index * 2 + 1, q_left, q_right));
        }
        if (q_right > mid) {
            res = max(res, query_interval(index * 2 + 2, q_left, q_right));
        }
        
        return res;
    }
    
public:
    SegmentTree(int n) : size(n) {
        tree.resize(4 * n);
        build(0, 0, n - 1);
    }
    
    void update(int left, int right, int val) {
        update_interval(0, left, right, val);
    }
    
    int query(int left, int right) {
        return query_interval(0, left, right);
    }
};

class Solution {
public:
    int busyStudent(vector<int>& startTime, vector<int>& endTime, int queryTime) {
        SegmentTree segTree(1010);
        int size = startTime.size();
        for (int i = 0; i < size; ++i) {
            segTree.update(startTime[i], endTime[i], 1);
        }
        return segTree.query(queryTime, queryTime);
    }
};
```

**思路 3：树状数组**
因为 ，所以我们可以维护一个区间为  的树状数组。
注意：
1.树状数组中  指的是将对应元素  加上 。
2.: 指的是  位置之前的元素和，即前缀和。
然后遍历所有学生的开始时间和结束时间，将树状数组上  的值增加 ，再将树状数组上  的值减少 。
则查询  位置的前缀和即为答案。
时间复杂度：，其中  为数组元素的个数。
空间复杂度：。
**思路3代码：**

```c++
class BinaryIndexTree {
private:
    vector<int> tree;
    int size;

    int lowbit(int index) {
        return index & (-index);
    }

public:
    BinaryIndexTree(int n) : size(n), tree(n + 1, 0) {}

    void update(int index, int delta) {
        while (index <= size) {
            tree[index] += delta;
            index += lowbit(index);
        }
    }

    int query(int index) {
        int res = 0;
        while (index > 0) {
            res += tree[index];
            index -= lowbit(index);
        }
        return res;
    }
};

class Solution {
public:
    int busyStudent(vector<int>& startTime, vector<int>& endTime, int queryTime) {
        BinaryIndexTree bit(1010);
        int size = startTime.size();
        for (int i = 0; i < size; ++i) {
            bit.update(startTime[i], 1);
            bit.update(endTime[i] + 1, -1);
        }
        return bit.query(queryTime);
    }
};
```

### 1620. 网络信号最好的坐标

**题目描述：** 给你一个数组 towers 和一个整数 radius ，数组中包含一些网络信号塔，其中 towers[i] = [xi, yi, qi] 表示第 i 个网络信号塔的坐标 (xi, yi) 和信号强度 qi 。所有坐标都是在 X-Y 坐标系内的整数坐标。两个坐标之间的距离用 欧几里得距离 计算。

**题目要求：** 计算在所有可能的坐标中，信号强度最大且距离原点 (0, 0) 最近的坐标。如果有多个这样的坐标，返回距离原点最近的坐标。

**说明：**

> 

- 

- 

- 

- 

**示例1：**

```plaintext
输入：towers = [[1,2,5],[2,1,7],[3,1,9]], radius = 2
输出：[2,1]
解释：
- 坐标 (2, 1) 的信号强度为 13
  - 距离为 1 的塔的信号强度为 7 
  - 距离为 1 的塔的信号强度为 5 
  - 距离为 2 的塔的信号强度为 7
  - 距离为 2 的塔的信号强度为 5
  - 距离为 2 的塔的信号强度为 9
```

**思路1解析：** 暴力法，两层循环遍历数组，找到和为target的两个数。时间复杂度，空间复杂度。
**思路1代码：**

```c++
class Solution {
public:
    vector<int> bestCoordinate(vector<vector<int>>& towers, int radius) {
        int maxSignal = 0;
        vector<int> bestCoord(2, 0);
        for (int x = 0; x <= 50; x++) {
            for (int y = 0; y <= 50; y++) {
                int signal = 0;
                for (const auto& tower : towers) {
                    int dx = tower[0] - x;
                    int dy = tower[1] - y;
                    double distance = sqrt(dx * dx + dy * dy);
                    if (distance <= radius) {
                        signal += floor(tower[2] / (1 + distance));
                    }
                }
                if (signal > maxSignal) {
                    maxSignal = signal;
                    bestCoord[0] = x;
                    bestCoord[1] = y;
                }
            }
        }
        return bestCoord;
    }
};
```

### 2427. 公因子的数目

**题目描述：** 给你两个正整数 a 和 b ，返回 a 和 b 的公因子的数目。
**题目要求：** 公因子是指能同时整除 a 和 b 的正整数。

**说明：**

> 

- 

**示例1：**

```plaintext
输入：a = 12, b = 6
输出：4
```

**示例2：**

```plaintext
输入：a = 25, b = 30
输出：2
```

**思路1解析：** 暴力法，两层循环遍历数组，找到和为target的两个数。时间复杂度，空间复杂度。
**思路1代码：**

```c++
class Solution {
public:
    int commonFactors(int a, int b) {
        int count = 0;
        for (int i = 1; i <= min(a, b); i++) {
            if (a % i == 0 && b % i == 0) {
                count++;
            }
        }
        return count;
    }
};          
```

### 剑指 Offer 57 - II. 和为s的连续正数序列

**题目描述：** 输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。
**题目要求：** 序列内按照从小至大的顺序，序列间按照开始数字从小到大的顺序。

**说明：**

> 

- 

**示例1：**

```plaintext
输入：target = 9
输出：[[2,3,4],[4,5]]
```

**思路1解析：** 暴力法，两层循环遍历数组，找到和为target的两个数。时间复杂度，空间复杂度。
**思路1代码：**

```c++
class Solution {
public:
    vector<vector<int>> findContinuousSequence(int target) {
        vector<vector<int>> result;
        for (int i = 1; i <= target / 2; i++) {
            int sum = 0;
            vector<int> sequence;
            for (int j = i; j <= target; j++) {
                sum += j;
                sequence.push_back(j);
                if (sum == target) {
                    result.push_back(sequence);
                    break;
                }
                if (sum > target) {
                    break;
                }
            }   
        }
        return result;
    }
};
```

### 2249. 统计圆内格点数目

**题目描述：** 给你一个二维整数数组 circles ，其中 circles[i] = [xi, yi, ri] 表示一个圆，其中 (xi, yi) 是圆的圆心，ri 是圆的半径。
**题目要求：** 返回出现在至少一个圆内的格点数目。

**说明：**

> 

- 

- 

- 

- 

**示例1：**

```plaintext
输入：circles = [[1,1,1],[2,2,2],[3,3,3]]
输出：16
```

**思路1解析：** 暴力法，两层循环遍历数组，找到和为target的两个数。时间复杂度，空间复杂度。
**思路1代码：**

```c++
class Solution {
public:
    int countLatticePoints(vector<vector<int>>& circles) {
        int count = 0;
        for (int x = 0; x <= 100; x++) {
            for (int y = 0; y <= 100; y++) {
                for (const auto& circle : circles) {
                    int dx = x - circle[0];
                    int dy = y - circle[1]; 
                    if (dx * dx + dy * dy <= circle[2] * circle[2]) {
                        count++;
                        break;
                    }
                }
            }
        }
        return count;
    }
};
```

### 0800. 相似 RGB 颜色

**题目描述：** 如果一个颜色可以通过将另一个颜色每个分量四舍五入到最接近的整数得到，那么这两个颜色是相似的。
**题目要求：** 给定一个颜色，返回一个与它相似的 RGB 颜色，形式为 “#RRGGBB” 。

**说明：**

> 

- 颜色由六个十六进制数字组成，前两位表示红色，中间两位表示绿色，最后两位表示蓝色。

- 所有颜色都是 RGB 颜色。

**示例1：**

```plaintext
输入：color = "#09f166"
输出："#11ee66"
解释：
- 给定颜色 "#09f166" 的相似颜色可以是 "#11ee66" 。
```

**思路1解析：** 暴力法，两层循环遍历数组，找到和为target的两个数。时间复杂度，空间复杂度。
**思路1代码：**

```c++
class Solution {
public:
    string similarRGB(string color) {
        return "#" + closestHex(color.substr(1, 2)) + closestHex(color.substr(3, 2)) + closestHex(color.substr(5, 2));
    }

    string closestHex(const string& component) {
        int value = stoi(component, nullptr, 16);
        int closest = value / 17 * 17;  
        return (closest < 10 ? "0" : "") + to_string(closest) + (closest + 17 > 255 ? "" : "1");
    }
};
```

### 0221. 最大正方形

**题目描述：** 在一个由 ‘0’ 和 ‘1’ 组成的二维矩阵内，找到只包含 ‘1’ 的最大正方形，并返回其面积。
**题目要求：** 

**说明：**

> 

- 

- 

- 

- 为或

**示例1：**

```plaintext
输入：matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
输出：4
```

**示例2：**

```plaintext
输入：matrix = [["0","1"],["1","0"]]
输出：1
```

**思路1解析：** 暴力法，两层循环遍历数组，找到和为target的两个数。时间复杂度，空间复杂度。
**思路1代码：**

```c++
class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        int maxSide = 0;
        for (int i = 0; i < matrix.size(); i++) {
            for (int j = 0; j < matrix[i].size(); j++) {
                if (matrix[i][j] == '1') {
                    maxSide = max(maxSide, 1);  
                    if (i > 0 && j > 0) {
                        int side = min(min(dp[i-1][j], dp[i][j-1]), dp[i-1][j-1]) + 1;
                        dp[i][j] = side;
                        maxSide = max(maxSide, side);
                    }
                }
            }
        }   
        return maxSide * maxSide;
    }
};
```

### 0560. 和为 K 的子数组

**题目描述：** 给你一个整数数组 nums 和一个整数 k ，请你统计并返回 该数组中和为 k 的连续子数组的个数 。
**题目要求：** 

**说明：**

> 

- 

- 

- 

**示例1：**

```plaintext
输入：nums = [1,1,1], k = 2
输出：2
```

**示例2：**

```plaintext
输入：nums = [1,2,3], k = 3
输出：2
```

**思路1解析：** 暴力法，两层循环遍历数组，找到和为target的两个数。时间复杂度，空间复杂度。
**思路1代码：**

```c++
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        int count = 0;
        int sum = 0;
        unordered_map<int, int> prefixSumCount;
        prefixSumCount[0] = 1;
        for (int num : nums) {
            sum += num; 
            if (prefixSumCount.find(sum - k) != prefixSumCount.end()) {
                count += prefixSumCount[sum - k];
            }
            prefixSumCount[sum]++;
        }
        return count;
    }
};  
```

**思路2解析：** 前缀和+哈希表，时间复杂度，空间复杂度。
**思路2代码：**

```c++
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        int count = 0;
        int sum = 0;
        unordered_map<int, int> prefixSumCount;
        prefixSumCount[0] = 1;
        for (int num : nums) {
            sum += num;     
            if (prefixSumCount.find(sum - k) != prefixSumCount.end()) {
                count += prefixSumCount[sum - k];
            }
            prefixSumCount[sum]++;
        }
        return count;
    }
};
```
