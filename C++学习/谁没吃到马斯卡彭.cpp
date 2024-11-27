#include<iostream>
using namespace std;

int main(){
    int n,m;
    cin>>n>>m;
    int a[n+1];
    for(int i=1;i<=n;i++){
        a[i]=0;
    }
    int i=0,j=0;
    while(j<n-1){
        i++;
        if(a[i%n]==0){
            if(i%m==0){
                a[i%n]=1;
                j++;
            }
        }
    }
    int x;
    i=1;
    while(1){
        if(a[i++]==0){
            x=i-1;
            break;
        }
    }
    cout<<x<<endl;
    return 0;
}