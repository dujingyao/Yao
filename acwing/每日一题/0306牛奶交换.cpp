#include<bits/stdc++.h>
using namespace std;

typedef long long ll;
const int N=2e5+10;
ll a[N],b[N];
ll n,m,res;
string s;

int main(){
    scanf("%d %d",&n,&m);
    cin>>s;//字符串的长度和奶牛的个数一样
    for(int i=0;i<n;i++) scanf("%d",&a[i]),b[i]=a[i],res+=a[i];
    while(m--){
        for(int i=0;i<n;i++){
            if(a[i]==0) continue;
            if(i==0){
                if(s[0]=='L') a[n-1]++,a[0]--;
                if(s[0]=='R') a[0]--,a[1]++;
                continue;
            }
            if(i==n-1){
                if(s[n-1]=='L') a[n-1]--,a[n-2]++;
                if(s[n-1]=='R') a[n-1]--,a[0]++;
                continue;
            }
            if(s[i]=='L') a[i]--,a[i-1]++;
            if(s[i]=='R') a[i]--,a[i+1]++;

        }
        for(int i=0;i<n;i++){
            if(a[i]>b[i]){
                res-=a[i]-b[i];
                a[i]=b[i];
            }
        }
    }
    cout<<res<<endl;

    return 0;
}