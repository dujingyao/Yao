#include<iostream>
#include<cstring>
#include<algorithm>
using namespace std;
const int N=210,INF=1e9;
int n,m,k;
int x,y,z;
int d[N][N];
void floyd(){
    for(int k=1;k<=n;k++){
        for(int i=1;i<=n;i++){
            for(int j=1;j<=n;j++){
                d[i][j]=min(d[i][j],d[i][k]+d[k][j]);
            }
        }
    }
}
int main(){
    cin>>n>>m>>k;
    //输入邻接矩阵
    for(int i=1;i<=n;i++){
        for(int j=1;j<=n;j++){
            if(i==j) d[i][j]=0;
            else d[i][j]=INF;
        }
    }
    while(m--){
        cin>>x>>y>>z;
        d[x][y]=min(d[x][y],z);
        //可能存在多条路,保存最短路
    }
    floyd();
    while(k--){
        cin>>x>>y;
        if(d[x][y]>INF/2) puts("impossible");
        else cout<<d[x][y]<<endl;
    }
    return 0;
}