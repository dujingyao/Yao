#include<iostream>
#include<algorithm>
#include<cstring>
using namespace std;

const int N=1000;

int n,m;

bool st[N];

//单边无向图
struct g{
    int a;
    int b;
    int w;
}g[N][N];

//深度优先遍历
void dfs(int x){
    if(st[x]){
        return;
    }
    cout<<x<<' ';
    st[x]=true;
    for(int i=1;i<=n;i++){
        if(g[x][i].w!=-1){//说明有边
            dfs(i);
        }
    }
}

int main(){
    cin>>n>>m;
    for(int i=1;i<=n;i++){
        for(int j=1;j<=n;j++){
            g[i][j].w=-1;
        }
    }
    while(m--){
        int a,b,w;
        cin>>a>>b>>w;
        g[a][b].w=w;
        g[b][a].w=w;
    }
    dfs(1);
    return 0;
}