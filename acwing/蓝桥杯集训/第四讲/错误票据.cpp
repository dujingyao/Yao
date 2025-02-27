#include<bits/stdc++.h>
using namespace std;

const int N=1e5+10,INF=0x3f3f3f3f;
int a[N];

int main(){
    
    int n;
    cin>>n;
    int minv=INF,maxv=-INF;
    int tp;
    while(cin>>tp){//直接读到文件结尾结束
        if(tp<minv) minv=tp;
        if(tp>maxv) maxv=tp;
        a[tp]++; 
    }
    int ans1=0,ans2=0;
    for(int i=minv;i<=maxv;i++){
        if(a[i]==0) ans1=i;
        if(a[i]==2) ans2=i;
    }
    cout<<ans1<<' '<<ans2<<endl;

    return 0;
}