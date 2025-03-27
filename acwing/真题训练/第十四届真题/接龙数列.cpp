#include<bits/stdc++.h>
using namespace std;
int dp[10];
int main(){
    int n,mx=0;
    cin>>n;
    for(int i=1;i<=n;i++){
        string ch;
        cin>>ch;
        int a=ch[0]-'0',b=ch.back()-'0';
        
        dp[b]=max(dp[b],dp[a]+1);
        mx=max(mx,dp[b]);
    }
    cout<<n-mx<<endl;
    return 0;
}