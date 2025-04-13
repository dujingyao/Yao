#include<bits/stdc++.h>
using namespace std;

const int N=1e5+10;

int a[N],tr[N];

int lowbit(int x){
    return x&-x;
}
//在第x位上加y
int n;
int add(int x,int y){
    for(int i=x;i<=n;i+=lowbit(i)) tr[i]+=y;
}
//求前缀和
int query(int x){
    int res=0;
    for(int i=x;i;i-=lowbit(i)) res+=tr[i];
    return res;
}

int main(){
    int m;
    cin>>n>>m;
    for(int i=1;i<=n;i++) cin>>a[i];
    //前缀和
    for(int i=1;i<=n;i++) add(i,a[i]);
    while(m--){
        int k,x,y;
        cin>>k>>x>>y;
        if(k==0) cout<<query(y)-query(x-1)<<endl;//求前缀和
        else add(x,y);//在x位上加y
    }
    
    return 0;
}