#include<iostream>
using namespace std;

int main(){
    int n;
    cin>>n;
    int x[50];
    x[1]=0;
    x[2]=1;
    if(n==1) cout<<x[1];
    if(n==2) cout<<x[1]<<' '<<x[2];
    if(n>2){
        cout<<x[1]<<' '<<x[2]<<' ';
        for(int i=3;i<=n;i++){
            x[i]=x[i-1]+x[i-2];
            cout<<x[i]<<' ';
        }
    }
    return 0;
}