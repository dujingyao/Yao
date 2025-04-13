#include<bits/stdc++.h>
using namespace std;

const int N=1e5+10;
int e[N],n,res=0;

int main(){
    cin>>n;
    char r[N];
    cin>>r+1;
    int h1=-1,he=-1,g1=-1,ge=-1;
    for(int i=1;i<=n;i++){
        cin>>e[i];
        if(h1==-1 && r[i]=='H') h1=i;
        if(g1==-1 && r[i]=='G') g1=i;
        if(r[i]=='H') he=i;
        if(r[i]=='G') ge=i;
    }
    if(h1==-1||g1==-1){
        cout<<0<<endl;
        return 0;
    }
    // 判断满足包含全部的牛的名单
    // 两种情况
    // 1. 一头领导者领导所属所有牛，另一头领导者领导另一个领导者
    // 2. 两个领导者分别领导自己的所有牛
    if(he<=e[h1]&&ge<=e[g1]) res++;
    for(int i=1;i<=h1;i++){
        if(r[i]=='G'&&e[i]>=h1) res++;
    }
    for(int i=1;i<=g1;i++){
        if(r[i]=='H'&&e[i]>=g1) res++;
    }
    cout<<res<<endl;
    return 0;
}