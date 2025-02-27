#include<bits/stdc++.h>
using namespace std;

const int N=1010;
int v[N],w[N];
int f[N][N];//只从前i个物品中选，总体积不小于j
int n,m;

int main(){

    cin>>n>>m;
    for(int i=1;i<=n;i++) cin>>v[i]>>w[i];
    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++){
            if(v[i]<=j){  //可以选i
                f[i][j]=max(f[i-1][j],f[i-1][j-v[i]]+w[i]);
            }
            else{
                f[i][j]=f[i-1][j];
            }
        }
    }
    cout<<f[n][m]<<endl;

    return 0;
}