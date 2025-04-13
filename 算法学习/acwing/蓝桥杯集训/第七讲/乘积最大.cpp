#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=100010,mod=1000000009;
int a[N];
int main(){
    int n,k;
    scanf("%d%d",&n,&k);
    for(int i=0;i<n;i++) scanf("%d",&a[i]);
    sort(a,a+n);
    ll res=1;//乘积初始化
    int l=0,r=n-1;//双指针初始化
    int sign=1;//符号初始化
    if(k%2){//如果k是奇数
        res=a[r];//取出最大的一个数
        r--;
        k--;//现在k是偶数了，可以按照偶数的方法来做了
        //如果连最大值都是负数，就证明全是负数
        if(res<0) sign=-1;
    }
    while(k){
        ll x=(ll)a[l]*a[l+1],y=(ll)a[r]*a[r-1];
        if(x*sign>y*sign){//取最大的一对
            res=x%mod*res%mod;
            l+=2;
        }
        else{
            res=y%mod*res%mod;
            r-=2;
        }
        k-=2;
    }
    printf("%lld",res);
    return 0;
}