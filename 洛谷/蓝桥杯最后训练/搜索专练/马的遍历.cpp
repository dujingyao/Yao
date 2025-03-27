#include<bits/stdc++.h>
using namespace std;
const int N=410;
typedef pair<int,int> PII;
queue<PII> q;
int dx[8]={2,2,-2,-2,1,1,-1,-1};
int dy[8]={1,-1,1,-1,2,-2,2,-2};
int n,m,x,y;
bool st[N][N];//标记是否走过
int dis[N][N];
int cnt;
void bfs(int x,int y){
    st[x][y]=true;
    q.push({x,y});
    dis[x][y]=0;
    //加入就判断是否非空
    while(!q.empty()){
        auto h=q.front();
        q.pop();
        for(int i=0;i<8;i++){
            //遍历每个方向
            int nowx,nowy;
            nowx=h.first+dx[i];
            nowy=h.second+dy[i];
            if(nowx>n||nowx<1) continue;
            if(nowy>m||nowy<1) continue;        
            if(st[nowx][nowy]) continue;
            //满足条件，加入
            dis[nowx][nowy]=dis[h.first][h.second]+1;
            st[nowx][nowy]=true;
            q.push({nowx,nowy});
        }
    }
}

int main(){
    cin>>n>>m>>x>>y;
    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++){
            dis[i][j]=-1;
        }
    }
    bfs(x,y);
    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++){
            cout<<dis[i][j]<<' ';
        }
        cout<<endl;
    }
    return 0;
}