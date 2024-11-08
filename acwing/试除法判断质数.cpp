#include<iostream>
#include<algorithm>
using namespace std;
bool check(int n){
    if(n==1) return false;
    if(n==2) return true;
    for(int i=2;i<=n/i;i++){
        if(n%i==0) return false;
    }
    return true;
}

int main(){
    int n;
    cin>>n;
    int a[n];
    for(int i=0;i<n;i++){
        cin>>a[i];
    }
    for(int i=0;i<n;i++){
        if(check(a[i])) cout<<"Yes"<<endl;
        else cout<<"No"<<endl;
    }

    return 0;
}