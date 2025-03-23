#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=5e5+10;
string a;
//记录所有首位字母的位置
ll s[N];//记录i前有几个tt
int main(){
    int n;
    ll res=0;
    cin>>n;
    int i=0;
    cin>>a;
    char x,y;
    cin>>x>>y;
    int up=0;
    //前缀和
    for(int i=0;i<a.size();i++){
        if(a[i]==x){
            res++;
        }     
        s[i]=res;
    }
    ll ans=0;
    for(int i=n-1;i<a.size();i++){
        if(a[i]==y){
            if(a[i-n+2]==x) ans+=s[i]-(s[i]-s[i-n+1]);
            else ans+=s[i]-(s[i]-s[i-n+2]);
        }
    }

    cout<<ans<<endl;
    return 0;
}