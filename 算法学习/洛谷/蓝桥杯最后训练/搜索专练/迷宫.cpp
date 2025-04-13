#include<bits/stdc++.h>
using namespace std;
const int N=10;
int dx[4]={1,-1,0,0},dy[4]={0,0,-1,1};
int n,m,t,ans;
bool st[N][N];
int startx,starty,endx1,endy1;
void dfs(int x,int y){
    if(x==endx1&&y==endy1){
        ans++;
        return;
    }
    for(int i=0;i<4;i++){
        int a=x+dx[i],b=y+dy[i];
        if(a>n||a<1||b>m||b<1) continue;
        if(st[a][b]) continue;
        st[a][b]=true;
        dfs(a,b);
        st[a][b]=false;
    }
}
int main(){
    cin>>n>>m>>t;
    int x,y;
    cin>>startx>>starty>>endx1>>endy1;
    for(int i=1;i<=t;i++){
        cin>>x>>y;
        st[x][y]=true;
    }    
    st[startx][starty]=true;
    dfs(startx,starty);
    cout<<ans<<endl;
    return 0;
}