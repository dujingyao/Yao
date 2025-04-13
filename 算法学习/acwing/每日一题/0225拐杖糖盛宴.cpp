#include<bits/stdc++.h>
using namespace std;

const int N=2e5+10;

typedef long long int ll;

ll a[N],b[N];

int n,m;

int main(){
    cin>>n>>m;
    for(int i=1;i<=n;i++) cin>>a[i];
    for(int i=1;i<=m;i++) cin>>b[i];
    for(int i=1;i<=m;i++){
        ll x=0;//距离地面的高度
        for(int j=1;j<=n;j++){
            if(x>=b[i]) break;
            if(a[j]<=x) continue;//够不到
            ll dalta=min(a[j],b[i])-x;//吃的部分
            a[j]+=dalta;
            x+=dalta;
            if(x>=b[i]) break;
        }
    }
    for(int i=1;i<=n;i++){
        cout<<a[i]<<endl;
    }

    return 0;
}
