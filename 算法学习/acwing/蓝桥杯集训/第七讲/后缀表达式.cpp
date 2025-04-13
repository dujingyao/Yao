#include<bits/stdc++.h>
using namespace std;
//大的相加，小的相减
typedef long long ll;
const int N=1e5+10;

ll a[2*N];
int n,m;//只要有一个负号，那么所有数都可以减去

int main(){
    cin>>n>>m;
    for(int i=0;i<n+m+1;i++){
        cin>>a[i];
    }
    sort(a,a+m+n+1);//从小到大排序
    ll res=0;
    if(!m){
        for(int i=0;i<n+m+1;i++) res+=a[i];
        cout<<res<<endl;
        return 0;
    }
    //加上一个最大的数，减去一个最小的数
    res+=a[n+m];
    res-=a[0];
    m--;//负号的数目减1
    for(int i=1;i<n+m+1;i++) res+=abs(a[i]);
    cout<<res<<endl;
    return 0;
}