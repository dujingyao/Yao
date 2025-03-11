#include<bits/stdc++.h>
using namespace std;
const int N=1e5+10;
int h[N][N],d[N][N];//第i和j城市之间相差的距离
void add(int x,int y,int w){
    h[x][y]=w;
    h[y][x]=w;
}
int n,dmax=0;
bool st[N][N];//该点已经走过
void dfs(int u){
    if(u>n) return;
    for(int i=1;i<=n;i++){
        if(u==i) continue;
        if(!st[u][i]) st[u][i]=true;
    }
}
//找一个距离最大的两点，计算花费
int main(){
    cin>>n;
    for(int i=1;i<=n-1;i++){
        int x,y,w;
        add(x,y,w);
    }
    dfs(1);
    
    return 0;
}