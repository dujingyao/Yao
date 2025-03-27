#include<bits/stdc++.h>
using namespace std;
const int N=100;
int a[N];//记录被放置的位置 a[i]=j代表(i,j)被放置了一个
int b1[N];//记录列
int b2[N];//记录右对角线 x+y
int b3[N];//记录左对角线 x-y+15
int n;
int ans=0;
void dfs(int u){//u代表层数
    if(u>n){
        ans++;
        if(ans<=3){
            for(int i=1;i<=n;i++){
                cout<<a[i]<<' ';
            }
            cout<<endl;
        }
        return;
    }
    for(int i=1;i<=n;i++){
        if(b1[i]==0&&b2[u+i]==0&&b3[u-i+15]==0){
            a[u]=i; //记录放置的位置
            //占位
            b1[i]=1;
            b2[u+i]=1;
            b3[u-i+15]=1;
            dfs(u+1);//下一层递归
            //取消占位
            b1[i]=0;
            b2[u+i]=0;
            b3[u-i+15]=0;
        }
    }
}
int main(){
    cin>>n;
    dfs(1);//传入的是第几横排
    cout<<ans<<endl;
    return 0;
}