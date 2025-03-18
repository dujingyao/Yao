#include<bits/stdc++.h>
using namespace std;
//也就是每个子段都要大于x*len
//其中len为l~r的长度，即r-l+1
//如果小于就要舍弃其中一个使满足条件1

const int N=5e4+10;
int a[N],s[N];
int t,n,x;

int main(){
    cin>>t;
    while(t--){
        cin>>n;
        for(int i=1;i<=n;i++) cin>>a[i],s[i]+=s[i-1]+a[i];
        cin>>x;
        int res=0;
        for(int l=1;l<=n-1;l++){
            if(a[l]==100001) continue;
            for(int r=l+1;r<=n;r++){
                int len=r-l+1;
                int minn=l;
                if(a[r]==100001) continue;
                if(x*len<=s[r]-s[l-1]){
                    continue;
                }
                else{
                    for(int k=l;k<=r;k++){
                        if(a[minn]<a[k])  minn=k;
                    }
                    for(int i=minn;i<=n;i++){
                        s[i]-=a[minn];
                    }
                    a[minn]=100001;
                    res++;//不被选择的数量
                }
            }
        }
        cout<<res<<endl;
    }
    return 0;
}