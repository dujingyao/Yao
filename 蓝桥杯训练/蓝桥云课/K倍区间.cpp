#include<bits/stdc++.h>
using namespace std;
const int N=1e5+10;
int n,k;
long long x,s[N],b[N],ans;
int main(){
    cin>>n>>k;
    for(int i=1;i<=n;i++){
        cin>>x;
        s[i]+=(s[i-1]+x)%k;
        ans+=b[s[i]];//记录前面出现多少次
        b[s[i]]++;//次数加一
    }
    cout<<ans+b[0]<<endl;
    return 0;
}