#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=5e5+10;
//vi代表第i个数的值 初始值
//li代表i左边的值
//ri代表i右边的值
ll v[N],l[N],r[N];

void del(int x){
    //相当于删除操作
    r[l[x]]=r[x],l[r[x]]=l[x];
    v[l[x]]+=v[x],v[r[x]]+=v[x];
}
int main(){
    int n,k;
    cin>>n>>k;
    r[0]=1,l[n+1]=n;
    //小根堆
    //记录
    priority_queue<pair<ll,int>,vector<pair<ll,int>>,greater<pair<ll,int>>> h;
    for(int i=1;i<=n;i++){
        cin>>v[i];
        //前一项
        l[i]=i-1;
        //后一项
        r[i]=i+1;
        //刚开始是初始值和初始坐标
        h.push({v[i],i});
    }
    while(k--){
        auto p=h.top();
        h.pop();
        //说明已经被重新赋值了
        if(p.first!=v[p.second]) h.push({v[p.second],p.second}),k++;
        else del(p.second); 
    }
    int head=r[0];
    while(head!=n+1){
        cout<<v[head]<<" ";
        head=r[head];
    }
    return 0;
}