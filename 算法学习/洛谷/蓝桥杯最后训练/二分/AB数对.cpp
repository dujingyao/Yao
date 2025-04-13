#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=2e5+10;
ll a[N],c,ans;
int n;

int main(){
    cin>>n>>c;
    for(int i=1;i<=n;i++) cin>>a[i];
    sort(a+1,a+n+1);
    for(int i=1;i<=n;i++){
        ll res=c+a[i];
        int x=0,y=0;
        int l=1,r=n;
        int mid=(l+r)/2;
        //取最左侧的
        while(l<r){
            int mid=(l+r)/2;//左边界下取整
            if(a[mid]>=res) r=mid;
            else l=mid+1;
        }
        x=l;
        if(a[x]!=res) continue;//说明没有这个数
        l=1,r=n;
        //取最右侧的
        while(l<r){
            mid=(l+r+1)/2;//右边界上取整
            if(a[mid]<=res) l=mid;
            else r=mid-1;
        }
        y=l;
        ans+=y-x+1;
    }    
    cout<<ans;
    return 0;
}