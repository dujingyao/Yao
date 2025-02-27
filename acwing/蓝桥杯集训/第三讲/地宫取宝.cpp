#include<bits/stdc++.h>
using namespace std;

const int N=60, MOD = 1000000007;
//f[i][j][k][c]代表在(i,j)处有k个物品，且最大价值为c
//c为w[i][j]
int f[N][N][13][14];

int w[N][N];

int main(){
    int n,m,k;
    cin>>n>>m>>k;
    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++){
            cin>>w[i][j];
            w[i][j]++;
        }
    }
    //代表在(1,1)这个位置，有0个物品且最大价值为0的方案数为1
    f[1][1][0][0]=1;
    //代表在(1,1)这个位置，有1个物品且最大价值为w[1][1]的方案数为1
    f[1][1][1][w[1][1]]=1;

    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++){
            if(i==1&&j==1) continue;
            for(int u=0;u<=k;u++){
                for(int v=0;v<=13;v++){//价值
                    int &val=f[i][j][u][v];
                    val=(val+f[i-1][j][u][v])%MOD;//不选
                    val=(val+f[i][j-1][u][v])%MOD;//不选
                    //选  选的话首先要有一个，所以u>0
                    if(u>0&&v==w[i][j]){
                        for(int c=0;c<v;c++){
                            val=(val+f[i-1][j][u-1][c])%MOD;
                            val=(val+f[i][j-1][u-1][c])%MOD;
                        }
                    }
                }
            }
        }
    }
    int res=0;
    for(int i=0;i<=13;i++) res=(res+f[n][m][k][i])%MOD;
    cout<<res<<endl;

    return 0;
}