#include<iostream>
#include<cstring>
#include<queue>
using namespace std;
const int N=110;
const int M=110;
int a[N][M];   //记录地图
int b[N][M];   //记录距离
int n,m;
typedef pair<int,int> PII;
void bfs(int k,int l){  //起点
    queue<PII> q;
    q.push({k, l});
    b[0][0]=0;
    while(!q.empty()){  //非空
        PII start=q.front();
        q.pop();
        a[start.first][start.second]=1;
        int dx[4]={0,0,-1,1},dy[4]={-1,1,0,0};
        for(int i=0;i<4;i++){
            int x=start.first+dx[i],y=start.second+dy[i];
            if(a[x][y]==0){
                a[x][y]=1;
                b[x][y]=b[start.first][start.second]+1;
                q.push({x,y});
            }
        }
    }
    cout<<b[n][m];
}
int main(){
    memset(a, 1, sizeof(a));
    cin>>n>>m;
    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++){
            cin>>a[i][j];
        }
    }
    bfs(1,1);
    return 0;
}