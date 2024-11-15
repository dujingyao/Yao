#include<iostream>
using namespace std;

void qmi(int a,int k,int p){
    long long res=1%p;
    while(k){
        if(k&1) res=res*a%p;
        a=a*(long long)a%p;
        k>>=1;
    }
    cout<<res<<endl;
}

int main(){
    
    int n;
    cin>>n;
    while(n--){
        int a,k,p;
        cin>>a>>k>>p;
        qmi(a,k,p);
    }

    return 0;
}