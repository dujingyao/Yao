#include<iostream>
#include<unordered_map>
using namespace std;

const int mod=1e9+7;

int main(){
    int T;
    cin>>T;
    unordered_map<int,int> map;
    while(T--){
        int n;
        cin>>n;
        for(int i=2;i<=n/i;i++){
            while(n%i==0){
                map[i]++;
                n/=i;
            }
        }
        if(n>1) map[n]++;
    }
    long long res=1;
    for(auto iter=map.begin();iter!=map.end();iter++){
        res=res*(iter->second+1)%mod;//数论的公式
    }
    cout<<res;

    return 0;
}