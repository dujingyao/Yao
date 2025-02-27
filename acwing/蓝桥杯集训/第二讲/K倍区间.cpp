#include<bits/stdc++.h>
using namespace std;
const int N=1000010;
int a[N];
long long int s[N],cnt[N],res;
int main(){
    
    int n,k;
    cin>>n>>k;
    for(int i=1;i<=n;i++){
        cin>>a[i];
        s[i]=(s[i-1]+a[i])%k;
        res+=cnt[s[i]];
        cnt[s[i]]++;//判断s[i]出现了多少次
    }
    cout<<res+cnt[0]<<endl;
    return 0;
}