#include<bits/stdc++.h>
using namespace std;

typedef long long ll;
const int N=2e5+10;
ll a[N];//原数组
ll b[N];//一阶差分
ll c[N];//二阶差分

int main(){
    
    int n;
    cin>>n;
    for(int i=1;i<=n;i++) cin>>a[i];
    for(int i=1;i<=n;i++) b[i]=a[i]-a[i-1];
    for(int i=1;i<=n;i++) c[i]=b[i]-b[i-1];
    ll res=0;
    for(int i=1;i<=n;i++) res+=abs(c[i]);
    cout<<res<<endl;
    return 0;
}