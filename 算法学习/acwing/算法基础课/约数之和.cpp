#include<iostream>
#include<algorithm>
#include<unordered_map>
#include<vector>
using namespace std;

typedef long long LL;

const int N=110,mod=1e9+7;

int main(){
    int n;
    cin>>n;
    unordered_map<int,int> primes;
    while(n--){
        int x;
        cin>>x;
        for(int i=2;i<=x/i;i++){
            while(x%i==0){
                x/=i;
                primes[i]++;
            }
        }
        if(x>1) primes[x]++;
    }

    LL res=1;
    for(auto p:primes){
        //取出质因子和对应的幂
        LL a=p.first,b=p.second;  //a代表质因数，b代表指数
        LL t=1;
        //求这一部分的和
        while(b--){
            t=(t*a+1)%mod;
        }
        res=res*t%mod;
    }
    cout<<res<<endl;
    return 0;
}