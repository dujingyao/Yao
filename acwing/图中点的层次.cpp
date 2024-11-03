#include<iostream>
#include<cstring>
#include<algorithm>
#include<queue>
using namespace std;
const int N=1e5+10;
int n,m;
int e[N],ne[N],h[N],idx;
int g[N];//记录距离
int st[N];//记录有没有走过
//将b插入a后面
void add(int a,int b){
    e[idx]=b;
    ne[idx]=h[a];
    h[a]=idx++;
}
void bfs(){
    queue<int> q;
    q.push(1);
    g[1]=0;
    st[1]=1;
    while(!q.empty()){  //q非空
        int start=q.front();
        q.pop();
        for(int i=h[start];i!=-1;i=ne[i]){
            int j=e[i];//与i相接的点
            if(st[j]==0){   //最先遍历到的,即最小的
                g[j]=g[start]+1;
                q.push(j);
                st[j]=1;
            }
        }
    }
}
int main(){
    memset(h,-1,sizeof(h));
    memset(g,-1,sizeof(g));
    cin>>n>>m;
    int a,b;
    for(int i=1;i<=m;i++){
        cin>>a>>b;
        add(a,b);
    }
    bfs();
    cout << g[n];
    return 0;
}