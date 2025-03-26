#include<bits/stdc++.h>
using namespace std;
const int N=1e6+10;
int a[N];
int n,m;
int main(){
    cin>>n>>m;
    for(int i=1;i<=n;i++) cin>>a[i];
    while(m--){
        int x;
        cin>>x;
        int l=1,r=n;
        int mid=(l+r)/2;
        while(l<r){
            if(a[mid]<x) l=mid+1;
            else r=mid;
            mid=(l+r)/2;
        }
        if(a[l]==x) cout<<l<<' ';
        else cout<<-1<<' ';
    }
    return 0;
}