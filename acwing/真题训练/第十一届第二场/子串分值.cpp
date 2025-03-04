#include<bits/stdc++.h>
using namespace std;

const int N=100010;

typedef long long int ll;
//l代表左边,r代表右边,h代表每次更新的下标
int l[N],r[N],h[27];
char st[N];

ll ans;

int main(){
    
    scanf("%s",st+1);
    int len=strlen(st+1);
    //记录前一个的位置
    for(int i=1;i<=len;i++){
        int t=st[i]-'a';
        l[i]=h[t];
        h[t]=i;
    }
    //再重置h[i]为最后
    for(int i=0;i<26;i++){
        h[i]=len+1;
    }
    //标记后面的
    for(int i=len;i;i--){
        int t=st[i]-'a';
        r[i]=h[t];
        h[t]=i;
    }
    //相加
    for(int i=1;i<=len;i++){
        ans+=(ll)(i-l[i])*(r[i]-i);
    }
    cout<<ans<<endl;


    return 0;
}