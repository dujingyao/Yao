#include<bits/stdc++.h>
using namespace std;

const int N=2e5+10;
int t[N],c[N];
int n,q,v,s;

int main(){
    cin>>n>>q;
    for(int i=1;i<=n;i++){
        cin>>c[i];
    }
    for(int i=1;i<=n;i++){
        cin>>t[i];
        c[i]-=t[i];//算出时间差
    }
    sort(c+1,c+1+n);
    while(q--){
        cin>>v>>s;//s时刻起床，至少访问v个农场
        //用二分找到第一个比s大的
        int l=1,r=n;
        //s严格大于mid
        while(l<r){
            int mid=(l+r)/2;
            if(c[mid]<=s){
                l=mid+1;
            }else{//c[mid]>s
                r=mid;
            }
        }
        int tot;
        if(c[l]<=s) tot=0;
        else tot=n-l+1;
        if(tot>=v) cout<<"YES"<<endl;
        else cout<<"NO"<<endl;
    }
    return 0;
}