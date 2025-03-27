#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=110;
ll a[N],sum;
int n,t;
int main(){
    cin>>t;
    while(t--){
        cin>>n;
        for(int i=1;i<=n;i++){
            cin>>a[i];
            sum+=a[i];
        }
        if(sum%n==0) cout<<0<<endl;
        else cout<<1<<endl;
        sum=0;
    }
    return 0;
}