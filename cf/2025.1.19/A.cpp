#include<iostream>
using namespace std;

int t;

int fib(int a[]){
    a[3]=a[1]+a[2];
    if(a[4]==a[2]+a[3]&&a[5]==a[3]+a[4]) return 3;
    a[3]=a[4]-a[2];
    if(a[5]==a[3]+a[4]) return 2;
    if(a[3]==a[1]+a[2]) return 2;
    return 1;
}

int main(){
    int a[7];
    cin>>t;
    for(int i=1;i<=t;i++){
        cin>>a[1]>>a[2]>>a[4]>>a[5];
        cout<<fib(a)<<endl;
    }

    return 0;
}