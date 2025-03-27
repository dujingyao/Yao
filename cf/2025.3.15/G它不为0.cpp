#include<bits/stdc++.h>
using namespace std;
//变成二进制后，至少有一位全是1
// << 左移 代表乘以2
// >> 右移 代表除以2
//分析哪一位的1最多
const int N=2e5+10;
int t,n;
int l,r;
int a[N],s[N];
int main(){
    cin>>t;
    while(t--){
        memset(s,0,sizeof s);
        cin>>l>>r;
        int len=r-l+1;
        int x=l,maxj=0;
        for(int i=1;i<=len;i++){
            a[i]=x;
            x++;
            int j=1;
            while(a[i]>=1){
                if(a[i]&1) s[j]++;
                a[i]/=2;
                j++;
            }
            maxj=max(maxj,j);
        }
        int maxlen=0;
        for(int i=1;i<=maxj;i++){
            maxlen=max(s[i],maxlen);//最多的1
        }
        cout<<len-maxlen<<endl;
    }
    return 0;
}