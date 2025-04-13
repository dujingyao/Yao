#include<iostream>
using namespace std;
typedef long long LL;

int qmi(int a,int b,int m){
    LL res=1;
    a%=m;
    while(b){//m为指数，将分解为二进制的形式
        if(b&1) res=res*a%m;
        a=a*(LL)a%m; //累乘a
        b>>=1; //右移b
    }
    return res;
}
bool check(int a,int b){
    if(a%b!=0) return true;
    else return false;
}

int main(){
    int n;
    cin>>n;
    while(n--){
        int a,b;
        cin>>a>>b;
        if(check(a,b)){
            int m=qmi(a,b-2,b);
            cout<<m<<endl;
        }
        else cout<<"impossible"<<endl;
    }
    return 0;
}