#include<bits/stdc++.h>
using namespace std;

const int N=1e5+10;
int price[N];
int n;
//最优 
//短视

int main(){
    cin>>n;
    for(int i=1;i<=n;i++) cin>>price[i];
    int res=0;
    for(int i=1;i+1<=n;i++){
        int dt=price[i+1]-price[i];
        if(dt>0) res+=dt;
    }
    cout<<res<<endl;
    return 0;
}