#include<bits/stdc++.h>
using namespace std;
const int N=110;
struct goldx{
    double m,v;
    double per_v; // 单位重量的价值
}gold[N];

bool cmp(goldx a,goldx b){
    return a.per_v > b.per_v; // 按单位价值从大到小排序
}

int main(){
    double W; // 背包容量
    int n; // 物品数量
    cin >> n >> W; // 读入物品数量和背包容量
    
    for(int i=1; i<=n; i++){
        cin >> gold[i].m >> gold[i].v;
        gold[i].per_v = gold[i].v / gold[i].m; // 计算单位重量的价值
    }
    
    double total_value = 0; // 总价值
    double curr_weight = 0; // 当前重量
    
    sort(gold+1, gold+n+1, cmp);
    
    for(int i=1; i<=n; i++){
        if(curr_weight + gold[i].m <= W){
            // 可以完整装入
            curr_weight += gold[i].m;
            total_value += gold[i].v;
        }
        else{
            // 只能装入一部分
            double remain = W - curr_weight;
            total_value += remain * gold[i].per_v;
            break;
        }
    }
    
    printf("%.2lf\n", total_value);
    return 0;
}