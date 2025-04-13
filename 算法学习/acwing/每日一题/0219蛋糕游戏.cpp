#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

const int N=5*1e5+10;
long long int e,n,T,x;

int main(){
    cin>>T;
    while(T--){
        long long int a[N];
        long long int f[N];
        e=0;
        cin>>n;
        long long int m=n/2-1;
        //输入蛋糕大小
        for(int i=1;i<=n;i++){
            cin>>a[i];
            f[i]=a[i]+f[i-1];
        }
        for(int i=0;i<=m;i++){
            e=max(e,f[i]+f[n]-f[n+i-m]);
        }
        cout<<f[n]-e<<' '<<e<<endl;
    }
    
    return 0;
}