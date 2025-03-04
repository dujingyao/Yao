#include<bits/stdc++.h>
using namespace std;

const int N=100000;
typedef long long int ll;
ll a[N];

int main(){
    int n,res=0;
    cin>>n;
    for(int i=1;i<=n;i++){
        cin>>a[i];
    }
    for(int i=1;i<=n-1;i++){
        for(int j=i+1;j<=n;j++){
            if(a[i]>a[j]) res++;
        }
    }
    cout<<res<<endl;
    return 0;
}