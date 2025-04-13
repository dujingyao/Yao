#include<bits/stdc++.h>
using namespace std;
const int N=1e5+10;
int n,k;
int h[N],w[N],maxsize,max1;
bool check(int mid){
    int num=0;
    for(int i=1;i<=n;i++){
        num+=(w[i]/mid)*(h[i]/mid);
        if(num>=k) return true;
    }
    return false;
}
int main(){
    cin>>n>>k;
    for(int i=1;i<=n;i++){
        cin>>h[i]>>w[i];
    }
    //如果i*i没有办法平分，那么(i+1)*(i+1)更没有办法平分
    int l=1,r=1e5;
    while(l<r){
        // 边长
        // 如果l和r很大时，可能会爆int
        // (r-l)/2可以理解为偏移的量，然后再l+(r-l)/2
        // 需要向上取整，所以再加一
        //向上取整公式⌈a ÷ b⌉ = (a + b - 1) ÷ b
        int mid=l+(r-l+1>>1);//防止溢出 避免死循环
        if(check(mid)) l=mid;
        else r=mid-1;
    }
    cout<<l<<endl;
    return 0;
}