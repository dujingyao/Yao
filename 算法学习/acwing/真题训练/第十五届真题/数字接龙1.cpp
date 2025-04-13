#include<bits/stdc++.h>
using namespace std;
int n,k;
const int N=15;
int g[N][N];
bool st[N][N];
int a[N][N][N][N];
bool is_vaild;
bool check(int x,int y,int nx,int ny){
    if(a[nx][ny][x][y]||a[x][y][nx][ny]) return false;
    return true;
}
string ans="";
void dfs(int sx,int sy,int p,string path,int num){
    if(is_vaild) return;
    st[sx][sy]=true;
    //结束
    if(sx==n-1&&sy==n-1&&num==n*n){
        is_vaild=true;
        st[sx][sy]=true;
        ans=path;
        return;
    }
    int dx[8]={-1,-1,0,1,1,1,0,-1},dy[8]={0,1,1,1,0,-1,-1,-1};
    char op[9]="01234567";
    for(int i=0;i<8;i++){
        int nx=sx+dx[i];
        int ny=sy+dy[i];
        if(nx<0||nx>=n||ny<0||ny>=n) continue;
        if(g[nx][ny]!=(p+1)%k) continue;
        if(st[nx][ny]) continue;
        if(!check(sx,sy,nx,ny)) continue;
        st[nx][ny]=true;
        a[sx][sy][nx][ny]=1;
        a[nx][ny][sx][sy]=1;
        string temp=path;
        int p2=(p+1)%k;
        dfs(nx,ny,p2,path,num+1);
        st[nx][ny]=false;
        a[sx][sy][nx][ny]=0;
        a[nx][ny][sx][sy]=0;
        path=temp;
    }

}
int main(){
    cin>>n>>k;
    string start="";
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
        {
            cin>>g[i][j];
        }
    }
    dfs(0,0,0,start,1);
    if(ans!="")
    {
        cout<<ans<<endl;
    }
    else
    {
        cout<<-1<<endl;
    }
    return 0;
}