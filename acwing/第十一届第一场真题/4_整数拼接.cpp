#include<bits/stdc++.h>
using namespace std;

typedef long long int ll;

int main(){
    ll n,k;
    cin>>n>>k;
    ll a[n+10];
    for(int i=1;i<=n;i++){
        cin>>a[i];
    }
    ll res=0;
    for(int i=1;i<=n;i++){
        for(int j=1;j<=n;j++){
            if(i==j) continue;
            ll backup=a[j];
            ll num=0;
            while(backup>=1){
                backup/=10;
                num++;
            }
            if((10*num*a[i]+a[j])%k==0) res++;
        }
    }
    cout<<res<<endl;
    return 0;
}