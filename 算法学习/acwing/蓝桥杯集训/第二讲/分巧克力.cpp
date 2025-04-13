#include<bits/stdc++.h>
using namespace std;

const int N=1e5+10;

int h[N],w[N];

int n,k;

bool check(int a){
    int num=0;//记录边长被分为a的巧克力数
    for(int i=1;i<=n;i++){
        num+=(h[i]/a)*(w[i]/a);
        if(num>=k) return true;
    }
    return false;
}

int main(){
    
    cin>>n>>k;
    for(int i=1;i<=n;i++){
        cin>>h[i]>>w[i];
    }
    //最多两重循环
    int l=1,r=1e5;
    while(l<r){
        int mid=l+(r-l+1>>1);//防止溢出
        if(check(mid)) l=mid;
        else r=mid-1;
    }
    cout<<r;

    return 0;
}