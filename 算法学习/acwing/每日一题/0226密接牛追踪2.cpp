#include<bits/stdc++.h>
using namespace std;

const int N=3e5+10;
string s;
vector<int> v;//根据0分段

int main(){
    int n;
    cin>>n>>s;
    s=s+'0';
    int res=n,cnt=0;//res代表经历了多少夜
    for(int i=0;i<=n;i++){
        if(s[i]=='1') cnt++;  //代表1的个数
        else if(s[i]=='0'&&cnt){//有多少个1
            if(i==cnt) res=min(res,cnt-1);//从头到第i个都是1
            else if(i==n) res=min(res,cnt-1);//到最后一个了，且最后一个是0
            else res=min(res,(cnt-1)/2);
            v.emplace_back(cnt);
            cnt=0;
        }
    }
    int ans=0;
    for(auto it:v){
        //res*2+1代表每头初始感染的牛覆盖的长度
        //分子再加上res*2是因为要上取整
        ans+=(it+res*2)/(res*2+1);
    }
    cout<<ans<<endl;
    return 0;
}