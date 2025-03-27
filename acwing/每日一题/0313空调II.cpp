#include<bits/stdc++.h>
using namespace std;

const int N=30;
int s[N],t[N];//每头牛活动范围的首尾
int c[N];//第n头牛的降温需求
int a[N],b[N];//每台空调管理的范围
int p[N];//第i台空调的降温值
int mcost[N];//第i台空调的运行成本
const int MAX_POS = 100000; // 覆盖足够大的位置
int down[MAX_POS + 10];      // 记录每个位置的累计降温值
int n,m1;//n头牛,m个空调
long long res=0,minx=1e18;

void dfs(int u){
    if(u>m1){//所有空调均检查完毕
        bool valid=true;
        for(int k=1;k<=n;k++){
            for(int h=s[k];h<=t[k];h++){//在第几个栏里面
                if(down[h]<c[k]){
                    valid=false;//不满足需求
                    break;
                }
            }
            if(!valid) break;
        }
        if(valid&&res<minx) minx=res;
        return;
    }
    //不选当前空调
    dfs(u+1);
    //选当前空调
    for(int j=a[u];j<=b[u];j++){
        down[j]+=p[u];
    }
    res+=mcost[u];
    dfs(u+1);
    //恢复现场
    res-=mcost[u];
    for(int j=a[u];j<=b[u];j++){
        down[j]-=p[u];
    }
}


int main(){

    cin>>n>>m1;
    for(int i=1;i<=n;i++){
        cin>>s[i]>>t[i]>>c[i];
    }
    for(int i=1;i<=m1;i++){
        cin>>a[i]>>b[i]>>p[i]>>mcost[i];
    }
    memset(down, 0, sizeof(down)); // 显式初始化
    dfs(1);
    cout<<minx<<endl;
    return 0;
}