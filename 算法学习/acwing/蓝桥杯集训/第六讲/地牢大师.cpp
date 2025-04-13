#include<bits/stdc++.h>
using namespace std;

const int N=110;

typedef tuple<int,int,int> TIII; 

char g[N][N][N];

int l,r,c,x1,yy1,z1,x2,y2,z2;

int dx[6] = {1, -1, 0, 0, 0, 0};
int dy[6] = {0, 0, 1, -1, 0, 0};
int dz[6] = {0, 0, 0, 0, 1, -1};

int ans=0;

int bfs(int x,int y,int z){
    queue<TIII> q;
    q.push({x,y,z});
    int dist[N][N][N];//已经代表最小距离了
    memset(dist,-1,sizeof dist);
    dist[x][y][z]=0;
    while(!q.empty()){
        auto [cx,cy,cz]=q.front();
        q.pop();
        for(int i=0;i<6;i++){
            int nx=cx+dx[i];
            int ny=cy+dy[i];
            int nz=cz+dz[i];
            if (nx >= 0 && nx < l && ny >= 0 && ny < r && nz >= 0 && nz < c && g[nx][ny][nz] != '#' && dist[nx][ny][nz] == -1) {
                dist[nx][ny][nz] = dist[cx][cy][cz] + 1;
                if (g[nx][ny][nz] == 'E') return dist[nx][ny][nz];
                q.push({nx, ny, nz});
            }
        }
    }
    return -1;
}

int main(){

    while(cin>>l>>r>>c,l||r||c){
        for(int i=0;i<l;i++){//层
            for(int j=0;j<r;j++){//行
                for(int k=0;k<c;k++){//列
                    cin>>g[i][j][k];
                    if(g[i][j][k]=='S') x1=i,yy1=j,z1=k;
                    if(g[i][j][k]=='E') x2=i,y2=j,z2=k;
                }
            }
        }
        int res=bfs(x1,yy1,z1);
        if (res == -1) cout << "Trapped!" << endl;
        else cout << "Escaped in " << res << " minute(s)." << endl;
    }

    return 0;
}