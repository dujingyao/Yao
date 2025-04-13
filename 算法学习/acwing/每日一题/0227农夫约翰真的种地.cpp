#include<bits/stdc++.h>
using namespace std;
const int N=2e5+10;
typedef long long ll;
struct S{
    ll num,sr,k;//num为高度排名，sr为当前高度，k为每日增长的高度吧
}f[N];

//写一个排名
bool cmp(S a,S b){
    return a.num<b.num;
}

ll h[N],a[N],t[N];
//根据数值排名
int main(){
    int T,n;
    cin>>T;
    while(T--){
        cin>>n;
        for(int i=1;i<=n;i++) cin>>h[i];//当前高度
        for(int i=1;i<=n;i++) cin>>a[i];//生长高度
        for(int i=1;i<=n;i++) cin>>t[i];//排名
        for(int i=1;i<=n;i++) f[i]={t[i],h[i],a[i]};
        sort(f+1,f+n+1,cmp);
        //需要取一个交集范围
        int min1=0,max1=1e9,sign=0;
        if(n==1){
            cout<<0<<endl;
            continue;
        }
        for(int i=1;i<=n-1;i++){
            int j=0;
            if(f[i].k>f[i+1].k){
                j=(f[i+1].sr-f[i].sr)/(f[i].k-f[i+1].k);//默认下取整
                if(f[i+1].sr<f[i].sr){//此时j是负数，c++为向0取整
                    j--;
                }
                min1=max(min1,j+1);
            }
            else if(f[i].k==f[i+1].k){
                if(f[i].sr<=f[i+1].sr) sign=1;//必定无解
            }
            else if(f[i].k<f[i+1].k){
                j=(f[i].sr-f[i+1].sr+f[i+1].k-f[i].k-1)/(f[i+1].k-f[i].k);//向上取整
                max1=min(max1,j-1);
            }
        }
        if(min1>max1||sign==1) cout<<-1<<endl;
        else cout<<min1<<endl;
    }
    return 0;
}