#include<bits/stdc++.h>
#define y second
#define x first
using namespace std;

typedef pair<int,int> PII;

const int N=210;
int T,n,m;
char g[N][N];
int dis[N][N];//记录距离


int bfs(PII start,PII end){
    queue<PII> q;
    int dx[4]={1,-1,0,0},dy[4]={0,0,-1,1};
    memset(dis,-1,sizeof dis);
    dis[start.x][start.y]=0;
    q.push(start);
    while(q.size()){
        auto t=q.front();
        q.pop();
        for(int i=0;i<4;i++){
            int x=t.x+dx[i],y=t.y+dy[i];
            if(x<=0||x>n||y<=0||y>m) continue;
            if(g[x][y]=='#') continue;
            if(dis[x][y]!=-1) continue;//说明已经走过了
            dis[x][y]=dis[t.x][t.y]+1;
            if(end==make_pair(x,y)) return dis[x][y];
            q.push({x,y});
        }
    }
    return -1;
}

int main(){
    cin>>T;
    while(T--){
        cin>>n>>m;
        for(int i=1;i<=n;i++){
            for(int j=1;j<=m;j++){
                cin>>g[i][j];
            }
        }
        PII start,end;
        for(int i=1;i<=n;i++){
            for(int j=1;j<=m;j++){
                if(g[i][j]=='S') start={i,j};
                else if(g[i][j]=='E') end={i,j};
            }
        }
        int t=bfs(start,end);
        if(t==-1) cout<<"oop!"<<endl;
        else cout<<t<<endl;
    }
    return 0;
}