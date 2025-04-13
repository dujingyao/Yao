#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=2e5+10;
ll x,y,sum;
ll t,n,k,tem,sumx;
int main(){
    cin>>t;
    while(t--){
        cin>>n>>k;
        k--;
        tem=0;
        map<ll,ll> b;
        //可能有相同的
        for(int i=1;i<=n;i++){
            cin>>x>>y;
            b[x]+=y;
        }
        ll total=0;
        //hash按照人数标准来分组
        ll hash[4]=0;
        //const 只需要读取数据，防止意外修改数据
        //没有复制，只读引用
        for(const auto &entry:b){
            ll c=entry.second;
            if(c<3){//无法成组的多余的人
                total+=c;
            }else{//可以成组的
                total+=2;//每个a至少贡献两人
                c-=2;
                //分别计算每组
                hash[3]+=c/3;//3人一组的组数
                hash[c%3]+=1;//剩余1或2人的情况
            }

        }
        //尝试为k-1个队伍分配最多的人数
        //remain是需要分配的组
        ll remain=k;
        for(int i=3;i>0;i--){
            if(hash[i]<=remain){
                total+=i*hash[i];
                remain-=hash[i];
                hash[i]=0;
            }else{
                total+=i*hash[i];
                hash[i]-=remain;
                remain=0;
                break;
            }
        }

    }
    return 0;
}