#include<bits/stdc++.h>
using namespace std;

const int N=20;
string ans;
int g[N][N];
bool st[N][N];//判断该点有没有走过
int n,k,f=0;
int dx[8]={-1,-1,0,1,1,1,0,-1},dy[8]={0,1,1,1,0,-1,-1,-1};
bool a[N][N][N][N];//a[x1][y1][x2][y2]代表（x1,y1）（x2,y2）有条路

bool check(int x,int y,int cx,int cy){
    if (a[cx][y][x][cy] || a[x][cy][cx][y]) return false;
    return true;
}
void dfs(int x,int y){
    if(ans.size()==n*n-1&&x==n-1&&y==n-1){
        f=1;
        return;
    }
    st[x][y]=true;
    //dfs本质是栈
    for(int i=0;i<8;i++){
        int cx=x+dx[i],cy=y+dy[i];
        if(!st[cx][cy]&&cx>=0&&cx<n&&cy>=0&&cy<n&&g[cx][cy]==(g[x][y]+1)%k){  
            if(!check(x,y,cx,cy)) continue;
            a[x][y][cx][cy]=true;
            a[cx][cy][x][y]=true;
            ans+=i+'0';
            dfs(cx,cy);
            if(f==1) return;
            st[cx][cy]=false;
            ans.pop_back();
            a[x][y][cx][cy]=false;
            a[cx][cy][x][y]=false;
        }
    }
    st[x][y]=false;
}
int main(){
    cin>>n>>k;
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            cin>>g[i][j];
        }
    }
    dfs(0,0);
    if(f==1) cout<<ans<<endl;
    else cout<<-1<<endl;
    return 0;
}