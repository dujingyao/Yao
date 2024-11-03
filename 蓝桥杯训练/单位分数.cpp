#include<iostream>
#include<vector>
#include<algorithm>
#define MAXNUMBER 30
using namespace std;

vector<int> divisors(int a){
    vector<int> n;
    for(int i=1;i*i<a;i++){
        if(a%i==0){
            n.push_back(i);
        }
        if(i*i!=a){
            n.push_back(a/i);
        }
    }
    sort(n.begin(),n.end());
    return n;
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
void dfs(vector<int>& nums, int index, int count, vector<vector<int>>& result) {
    if (count == nums.size()) {
        if (checkSum(nums)) {
            result.push_back(nums);
        }
        return;
    }
    for (int i = index; i <= 30; i++) {
        nums[count] = i;
        dfs(nums, i + 1, count + 1, result);
    }
}
int main(){
     int n;
    cin >> n;
    vector<vector<int>> result;
    vector<int> nums(n);
    dfs(nums, 2, 0, 1, result);

    sort(result.begin(), result.end());

    for (auto& r : result) {
        for (auto& x : r) {
            cout << "1/" << x << " ";
        }
        cout << endl;
    }

    return 0;
    return 0;
}