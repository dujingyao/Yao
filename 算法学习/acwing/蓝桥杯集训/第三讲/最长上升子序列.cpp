#include<bits/stdc++.h>
using namespace std;

const int N=1010;
typedef long long ll;
ll a[N],f[N];//f[i]表示以i结尾的子序列的最大上升子序列

int main(){
    
    int n;
    cin>>n;
    ll m=-1e9-1;
    for(int i=1;i<=n;i++) cin>>a[i];
    for(int i=1;i<=n;i++){
        f[i]=1;
        for(int j=1;j<i;j++){
            if(a[i]>a[j]) f[i]=max(f[i],f[j]+1);
        }
        m=max(m,f[i]);
    }
    cout<<m<<endl;
    return 0;
}