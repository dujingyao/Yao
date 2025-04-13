#include <iostream>
#include <vector>
#include <cmath> // 用于 pow 函数

using namespace std;

// 定义一个函数，计算给定整数列表的最小平方和
long long calculateMinSquareSum(const vector<int>& nums) {
    vector<long long> bitCounts(32, 0); // 创建一个长度为32的长整型向量，用于记录每一位上1出现的次数

    // 对于列表中的每一个数
    for (int num : nums) {
        // 对于每一个比特位（从0到31）
        for (int i = 0; i < 32; ++i) {
            // 如果当前数的第i位是1
            if (num >> i & 1) {
                // 相应的bitCounts[i]自增1
                bitCounts[i]++;
            }
        }
    }

    long long result = 0; // 初始化结果变量为0
    long long halfSize = nums.size() / 2; // 计算列表长度的一半

    // 对于每一个比特位
    for (int i = 0; i < 32; ++i) {
        // 如果1的数目大于列表长度的一半
        if (bitCounts[i] > halfSize) {
            // 加上（列表长度的一半减去（列表长度减去1的数目））的平方
            result += pow(halfSize - (nums.size() - bitCounts[i]), 2);
        } else {
            // 否则加上1的数目的平方
            result += pow(bitCounts[i], 2);
        }
    }

    // 返回最终的结果
    return result;
}

int main() {
    vector<int> nums = {9226, 4690, 4873, 1285, 4624, 1596, 6982, 590, 8806, 121,
                        8399, 8526, 5426, 64, 9655, 7705, 3929, 3588, 7397, 8020,
                        1311, 5676, 3469, 2325, 1226, 8203, 9524, 3648, 5278, 8647};

    long long minSquareSum = calculateMinSquareSum(nums);

    cout << "The minimum square sum is: " << minSquareSum << endl;

    return 0;
}