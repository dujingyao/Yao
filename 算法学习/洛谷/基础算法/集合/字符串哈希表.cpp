#include<bits/stdc++.h>
using namespace std;

const int MOD=23333,base=261,max_n=1510;

int n,ans;
char s[max_n];
vector<string> linker[MOD+2];

//插入哈希表
inline void insert(){
    int hash=1;
    for(int i=0;s[i];i++){
        hash=(hash*111*base+s[i])%MOD;//计算出字符串的哈希值
    }
    string t=s;
    for(int i=0;i<linker[hash].size();i++){
        if(linker[hash][i]==t)//如果已经有一个了
            return;
    }
    linker[hash].push_back(t);
    ans++;
}

int main(){
    cin>>n;
    for(int i=1;i<=n;i++){
        cin>>s,insert();
    }
    cout<<ans<<endl;
    return 0;
}