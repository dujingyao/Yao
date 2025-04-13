#include<bits/stdc++.h>
using namespace std;

const int N=30;
int w,h,x,y;
char g[N][N];

int res;

int dx[4]={1,-1,0,0},dy[4]={0,0,-1,1};

void dfs(int x,int y){
    g[x][y]='#';
    res++;
    for(int i=0;i<4;i++){
        int a=x+dx[i],b=y+dy[i];
        if(a<=0||a>w||b<=0||b>h||g[a][b]=='#'){
            continue;
        }
        dfs(a,b);
    }
}

int main(){
    //前置输入
    while(cin>>h>>w,h||w){
        res=0;
        for(int i=1;i<=w;i++){
            for(int j=1;j<=h;j++){
                cin>>g[i][j];
                if(g[i][j]=='@') x=i,y=j;
                if(g[i][j]=='0'&&j==2&&g[i][j-1]=='0') break;
            }
        }
        dfs(x,y);
        cout<<res<<endl;
    }



    return 0;
}