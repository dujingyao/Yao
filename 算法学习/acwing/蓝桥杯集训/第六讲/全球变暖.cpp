#include<bits/stdc++.h>
using namespace std;

typedef pair<int,int> PII;
const int N=1010;
char g[N][N];
bool st[N][N];
int n,cnt=0;
PII q[N*N];//数组模拟队列
int dx[4]={0,0,1,-1},dy[4]={1,-1,0,0};
void bfs(int i,int j,int &total,int &bound){
    int hh=0,tt=0;//表示队头队尾序号
    q[0]={i,j};//放入队头
    st[i][j]=true;
    while(hh<=tt){
        auto a=q[hh++];
        total++;
        bool is_bound=false;//初始化边界标记
        for(int k=0;k<4;k++){
            int x=a.first+dx[k];
            int y=a.second+dy[k];
            if(x<0||x>=n||y<0||y>=n) continue;
            if(st[x][y]) continue;
            if(g[x][y]=='.'){//那么为边界
                is_bound=true;
                continue;
            }
            q[++tt]={x,y};
            st[x][y]=true;
        }
        if(is_bound) bound++;
    }
}

int main(){
    
    cin>>n;
    for(int i=0;i<n;i++) cin>>g[i];
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if(g[i][j]=='#'&&!st[i][j]){
               int total=0,bound=0;
               bfs(i,j,total,bound);
               if(total==bound) cnt++;
            }
        }
    }
    cout<<cnt<<endl;

    return 0;
}