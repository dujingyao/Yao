#include <iostream>
#include <vector>
#include <string>
using namespace std;

// 构建 next 数组
void get_next(const string& pattern, vector<int>& next) {
    int n = pattern.length();
    next.resize(n, 0);
    int j = 0;  // j 表示前缀末尾位置
    for (int i = 1; i < n; ++i) {
        while (j > 0 && pattern[i] != pattern[j]) {
            j = next[j - 1];  // 回退 j
        }
        if (pattern[i] == pattern[j]) {
            ++j;
        }
        next[i] = j;
    }
}

// 构建 nextval 数组
void get_nextval(const string& pattern, const vector<int>& next, vector<int>& nextval) {
    int n = pattern.length();
    nextval.resize(n, 0);
    nextval[0] = 0;
    for (int i = 1; i < n; ++i) {
        if (pattern[i] != pattern[next[i]]) {
            nextval[i] = next[i];
        } else {
            nextval[i] = nextval[next[i]];
        }
    }
}

int main() {
    string pattern;
    cout << "请输入一个字符串: ";
    cin >> pattern;
    vector<int> next;
    get_next(pattern, next);
    vector<int> nextval;
    get_nextval(pattern, next, nextval);
    cout << "next 数组: ";
    for (int i = 0; i < pattern.length(); ++i) {
        cout << next[i] << " ";
    }
    cout << endl;
    cout << "nextval 数组: ";
    for (int i = 0; i < pattern.length(); ++i) {
        cout << nextval[i] << " ";
    }
    cout << endl;
    return 0;
}