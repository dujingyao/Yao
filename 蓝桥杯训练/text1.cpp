#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

typedef long long ll;

vector<ll> divisors(ll n) {
    vector<ll> res;
    for (ll i = 1; i * i <= n; i++) {
        if (n % i == 0) {
            res.push_back(i);
            if (i * i != n) {
                res.push_back(n / i);
            }
        }
    }
    sort(res.begin(), res.end());
    return res;
}
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

typedef long long ll;

vector<ll> divisors(ll n) {
    vector<ll> res;
    for (ll i = 1; i * i <= n; i++) {
        if (n % i == 0) {
            res.push_back(i);
            if (i * i != n) {
                res.push_back(n / i);
            }
        }
    }
    sort(res.begin(), res.end());
    return res;
}

// Check if the sum of the reciprocals equals 1
bool checkSum(vector<int> &nums) {
    int lcm = nums;
    for (size_t i = 1; i < nums.size(); i++) {
        lcm = lcm * nums[i] / gcd(lcm, nums[i]);
    }
    int sum = 0;
    for (size_t i = 0; i < nums.size(); i++) {
        sum += lcm / nums[i];
    }
    return sum == lcm;
}


void dfs(vector<ll>& nums, int index, int count, ll prod, vector<vector<ll>>& result) {
    if (count == nums.size()) {
        if (prod == 1) {
            result.push_back(nums);
        }
        return;
    }
    for (int i = index; i <= 30; i++) {
        nums[count] = i;
        dfs(nums, i + 1, count + 1, prod * i / gcd(prod, i), result);
    }
}

int main() {
    int n;
    cin >> n;
    vector<vector<ll>> result;
    vector<ll> nums(n);
    dfs(nums, 2, 0, 1, result);

    sort(result.begin(), result.end());

    for (auto& r : result) {
        for (auto& x : r) {
            cout << "1/" << x << " ";
        }
        cout << endl;
    }

    return 0;
}