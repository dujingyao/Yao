#include<bits/stdc++.h>
using namespace std;

typedef long long LL; 

const int N=4e5+10;

int t,n,m;

LL a[N],s[N];

int main(){
    
    cin>>t;
    while(t--){
        cin>>n>>m;
        for(int i=1;i<=n;i++){
            cin>>a[i];
            a[i]%=m;
        }
        //破环，拓展
        for(int i=1;i<=n;i++){
            a[i+n]=a[i]+m;
        }
        //排序
        sort(a+1,a+2*n+1);
        //计算前缀和
        for(int i=1;i<=2*n;i++){
            s[i]=s[i-1]+a[i];
        }
        LL res=1e18;
        for(int i=1;i<=n;i++){
            int l=i,r=n+i-1;
            int mid=(l+r)>>1;
            LL sum=(2*mid-l-r)*a[mid]+(s[r]-s[mid])+(s[l-1]-s[mid-1]);//？
            res=min(res,sum);

        }
        cout<<res<<endl;

    }

    return 0;
}