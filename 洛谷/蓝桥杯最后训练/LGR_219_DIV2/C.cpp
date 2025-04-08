#include<bits/stdc++.h>
using namespace std;

typedef long long ll;
const int N=210;
ll a[N],b[N],c[N];
int n,m,k;

int main(){
    cin>>n>>m>>k;
    ll maxans=0;
    for(int i=1;i<=n;i++) cin>>a[i];
    for(int i=1;i<=m;i++) cin>>b[i];
    for(int i=1;i<=k;i++) cin>>c[i];
    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++){
            for(int h=1;h<=k;h++){
                maxans=max(maxans,a[i]*b[j]%c[h]);
            }
        }
    }
    
    cout<<maxans;
    return 0;
}