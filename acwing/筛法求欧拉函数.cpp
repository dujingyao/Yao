#include<iostream>
using namespace std;

const int N=1e6+10;

bool st[N];  //记录是否为合数
int primes[N],dex;   //素数表
int euler[N];//欧拉函数

void get_euler(int x){
    euler[1]=1;
    for(int i=2;i<=x;i++){
        if(!st[i]){//如果i是素数
            primes[dex++]=i;
            euler[i]=i-1;
        }
        for(int j=0;primes[j]<=x/i;j++){
            int t=primes[j]*i;  //t是一个合数
            st[t]=true;//登记t为合数
            if(i%primes[j]==0){//i为primes[j]的一个约数
                euler[t]=euler[i]*primes[j];
                break;
            }
            euler[t]=euler[i]*(primes[j]-1);
        }
    }
}

int main(){
    int n;
    cin>>n;
    get_euler(n);
    
    return 0;
}