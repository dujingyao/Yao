#include<iostream>
#include<vector>
#include<cmath>
using namespace std;
long long calculate(vector<int>& nums){
    vector<long long> bit(32);
    for(auto num:nums){
        for(int i=0;i<32;i++){
            if(num>>i&1){  //如果当前位数是1
                bit[i]++;
            }
        }
    }
    long long res=0;
    long long halfsize=nums.size()/2;
    for(int i=0; i<32;i++){
        if(bit[i]>halfsize){
            res+=pow(halfsize-(nums.size()-bit[i]),2);
        }else{
            res+=pow(bit[i],2);
        }
    }
    return res;
}
int main(){
    vector<int> nums={9226, 4690, 4873, 1285, 4624,
                596, 6982, 590, 8806, 121,
                8399, 8526, 5426, 64, 9655,
                7705, 3929, 3588, 7397, 8020,
                1311, 5676, 3469, 2325, 1226,
                8203, 9524, 3648, 5278, 8647};
    long long res=calculate(nums);
    return 0;
}