#include<bits/stdc++.h>
using namespace std;
const int N=15;
int plane[N][5];
bool st[N];//第i架飞机有没有降落
bool flag;
int T,n;

void dfs(int u,int time){
    if(u>=n){
        flag=true;
    }
    if(flag) return;
    for(int i=1;i<=n;i++){
        if(!st[i]){
            if(plane[i][2]<time) return;//剪枝
            st[i]=true;
            //当前时间小于飞机最早降落时间
            //那么应该从飞机最早降落时间开始计算
            if(time<plane[i][1]) dfs(u+1,plane[i][1]+plane[i][3]);
            //当前时间大于飞机降落最早时间
            //从当前时间开始计算
            else dfs(u+1,time+plane[i][3]);
            st[i]=false;
        }
    }
}

int main(){
    cin>>T;
    while(T--){
        for(int i=1;i<=n;i++) st[i]=false;
        cin>>n;
        memset(st,0,sizeof st);
        for(int i=1;i<=n;i++){
            for(int j=1;j<=3;j++){
                cin>>plane[i][j];
            }
            plane[i][2]+=plane[i][1];
        }
        flag=false;
        dfs(0,0);//降落0架飞机，0时刻
        if(flag) cout<<"YES"<<endl;
        else cout<<"NO"<<endl;
    }
    return 0;
}