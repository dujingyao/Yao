#include<bits/stdc++.h>
using namespace std;
const int N=1e5+10;
int a[N];//a[i]代表i出现了a[i]次
vector<int> b[N];//b[i]包含所有i的倍数的数
int n,m,x;
//最后转换为求最大公约数
int main(){
    cin>>n;
    for(int i=1;i<=n;i++){
        cin>>x;
        a[x]++;
        m=max(m,x);
    }
    for(int i=1;i<=m;i++){
        for(int j=i;j<=m;j+=i){
            if(a[j]){//i倍的
                for(int k=0;k<a[j];k++){
                    b[i].push_back(j);
                }
            }
        }
    }
    for(int i=m;i>=1;i--){
        if(b[i].size()>=3){
            sort(b[i].begin(),b[i].end());
            for(int j=0;j<3;j++){
                cout<<b[i][j]<<' ';
            }
            break;
        }
    }
    return 0;
}