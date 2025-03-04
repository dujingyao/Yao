#include<bits/stdc++.h>
using namespace std;

int main(){
    
    int n;
    cin>>n;
    long long int x,sum=0,res=0;
    for(int i=1;i<=n;i++){
        cin>>x;
        sum+=res*x;
        res+=x;
    }
    cout<<sum;
    return 0;
}