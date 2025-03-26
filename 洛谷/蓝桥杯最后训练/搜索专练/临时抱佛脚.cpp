#include<bits/stdc++.h>
using namespace std;
const int N=30;
int maxtime;//最大时间
int nowtime;//当前的时间
int sum;//时间和
int s[5],a[N];
int ans,nowdeep;//答案 当前有几科
void dfs(int x){
    if(x>nowdeep){
        maxtime=max(maxtime,nowtime);
        return;
    }
    //选这道题
    if(nowtime+a[x]<=sum/2){
        nowtime+=a[x];
        dfs(x+1);
        nowtime-=a[x];
    }
    //不选这道题
    dfs(x+1);
}
int main(){
    cin>>s[1]>>s[2]>>s[3]>>s[4];
    for(int i=1;i<=4;i++){
        sum=0;
        nowtime=0;
        nowdeep=s[i];
        for(int j=1;j<=nowdeep;j++){
            cin>>a[j];
            sum+=a[j];
        }
        maxtime=0;
        dfs(1);
        ans+=(sum-maxtime);
    }
    cout<<ans;
}