#include<iostream>
using namespace std;
const int N=1e6+10;
int n,dex=0,prime[N];
bool st[N];

void get(int n){
    for(int i=2;i<=n;i++){
        if(st[i]) continue;
        prime[dex++]=i;
        for(int j=i+i;j<=n;j+=i){
            st[j]=true;
        }
    }
}

//线性筛法
//st数组标记合数
void get2(int x){
    for(int i=2;i<=x;i++){
        if(!st[i]) prime[dex++]=i;
        for(int j=0;prime[j]<=x/i;j++){
            st[prime[j]*i]=true;
            if(i%prime[j]==0) break;
        }
    }
}
int main(){
    cin>>n;
    get2(n);
    cout<<dex<<endl;
    return 0;
}