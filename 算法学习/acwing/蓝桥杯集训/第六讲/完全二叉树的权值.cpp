#include<bits/stdc++.h>
using namespace std;

const int N=1e5+10;
int n;

int main(){
    cin>>n;
    int a[N];
    for(int i=1;i<=n;i++) cin>>a[i];
    long long res=LLONG_MIN;
    int ans=0;
    //i代表每一层开始的位置
    for(int d=1,i=1;i<=n;i*=2,d++){
        long long sum=0;
        //j代表每一层从第一个到最后一个的遍历
        for(int j=i;j<i*2&&j<=n;j++){
            sum+=a[j];
        }
        if(res<sum) res=sum,ans=d;
    }
    cout<<ans<<endl;
    return 0;
}