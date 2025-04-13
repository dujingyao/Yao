#include<bits/stdc++.h>
using namespace std;

typedef long long ll;
const int N=2e5+10;
ll a[N],b[N];
ll n,m,res;
string s;

int main(){
    cin>>n>>m;
    cin>>s;//字符串的长度和奶牛的个数一样
    for(int i=0;i<n;i++){
        cin>>a[i];
        res+=a[i];
    }
    ll ans=0;
    for(int i=0;i<n;i++){
        if(s[(i-1+n)%n]=='R'&&s[(i+1+n)%n]=='L'){
            if(s[i]=='R'){
                ll j=(i-1+n)%n;
                ll tmp=0;
                while(s[j]=='R'){
                    tmp+=a[j];
                    j=(j-1+n)%n;
                }
                ans+=min(tmp,m);
            }
            else{
                ll j=(i+1+n)%n;
                ll tmp=0;
                while(s[j]=='L'){
                    tmp+=a[j];
                    j=(j+1+n)%n;
                }
                ans+=min(tmp,m);
            }
        }
    }
    cout<<res-ans<<endl;

    return 0;
}