

## 动态规划基础模版
状态定义： 设 dp 为一维数组，其中 dp[i] 的值代表斐波那契数列的第 i 个数字。
转移方程： dp[i+1]=dp[i]+dp[i−1] ，即对应数列定义 f(n+1)=f(n)+f(n−1) 。
初始状态： dp[0]=1, dp[1]=1 ，即初始化前两个数字。
返回值： dp[n] ，即斐波那契数列的第 n 个数字。

作者：Krahets
链接：https://leetcode.cn/problems/climbing-stairs/solutions/2361764/70-pa-lou-ti-dong-tai-gui-hua-qing-xi-tu-ruwa/

爬梯子题解
https://leetcode.cn/problems/climbing-stairs/solutions/2361764/70-pa-lou-ti-dong-tai-gui-hua-qing-xi-tu-ruwa/?envType=study-plan-v2&envId=top-interview-150