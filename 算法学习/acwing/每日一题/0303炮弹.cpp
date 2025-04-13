#include<bits/stdc++.h>
using namespace std;
const int N=1e5+10;
int q[N],v[N],x[N];
int main(){
    int n,s,m=0;
    cin>>n>>s;
    for(int i=1;i<=n;i++){
        cin>>q[i]>>v[i];
        if(q[i]==1) m++;
    }
    //q=0为有跳板
    //q=1为有炮击目标
    int res=0,k=1,d=1;
    set<int> x;
    //第一次
    if(q[s]==1){//炮击
        if(k>=v[s]){
            res++;
            v[s]=-1;
        }
        s+=k;//位置
    }
    else{
        k+=v[s];
        d=-1;
        x[s]=k;
        s+=d*k;
    }
    if(s<1||s>n){
        cout<<res<<endl;
        return 0;
    }
    while(s>=1&&s<=n){
        if(q[s]==0){//说明有跳板
            k+=v[s];
            d=-d;
            if(x[s]==k) break;
            x[s]=k;
            s+=(d*k);
        }else{//说明是炮击
            if(k>=v[s]&&v[s]!=-1){
                res++;
                v[s]=-1;
            }
            s+=(d*k);
        }
        if(res==m) break;
        
    }
    cout<<res<<endl;
    return 0;
}