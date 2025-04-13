#include<bits/stdc++.h>
using namespace std;
const int N=1e4+10;
typedef long long ll;
typedef pair<ll,ll> PII;
PII m[N];
int main(){
    int n;
    cin>>n;
    for(int i=0;i<n;i++){
        int x,y;
        cin>>x>>y;
        m[i]={x,y};
    }
    //分别记录每个数的最小值
    ll minx=0;
    for(ll i=0;i<n;i++){
        for(ll j=m[i].first/(m[i].second+1);j<=m[i].first;j++){
            if(m[i].first-m[i].second*j<j){
                minx=max(minx,j);
                break;
            }
        }
    }
    //记录每个数的最大值
    ll maxx=1e9+1;
    for(int i=0;i<n;i++){
        ll f=m[i].first/m[i].second;
        maxx=min(maxx,f);
    }
    cout<<minx<<' '<<maxx<<endl;
    return 0;
}