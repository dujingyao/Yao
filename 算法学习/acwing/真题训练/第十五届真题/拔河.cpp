#include<bits/stdc++.h>
#define int long long
using namespace std;
const int N=1e3+10;
int n,js=0,a[N];
struct dui{
    int l,r,sum;
}d[N*N];
bool cmp(dui x,dui y){
    return x.sum<y.sum;
}
//判断是否包含
signed check(int x,int y,int a,int b){
    if((x>=a&&y<=b)||(x<=a&&y>=b)) return 0;
    else return 1;
}
signed main(){
    cin>>n;
    for(int i=1;i<=n;i++) cin>>a[i];
    for(int i=1;i<=n;i++){
        int s=0;
        for(int j=i;j<=n;j++){
            s+=a[j];
            d[js].sum=s;
            d[js].l=i;
            d[js].r=j;
            js++;
        }
    }
    sort(d,d+js,cmp);
    int ans=1e12+10;
    for(int i=1;i<js;i++){
        if(check(d[i].l,d[i].r,d[i-1].l,d[i-1].r))
            ans=min(ans,d[i].sum-d[i-1].sum);
    }
    cout<<ans;
    return 0;
}