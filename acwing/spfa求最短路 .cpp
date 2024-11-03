#include<iostream>
#include<algorithm>
#include<cstring>
using namespace std;
const int N=10e5+10;
int h[N],e[N],w[N],ne[N],idx;
int st[N];//标记定点是否在队列中
int dist[N];//保存最短路径的值
int q[N],hh,tt=-1;//队列
void add(int a,int b,int c){
    e[idx]=b;
    w[idx]=c;
    ne[idx]=h[a];
    h[a]=idx++;
}

void spfa(){
    q[++t]=1;
    dist[1]=0;
    st[1]=1;
    
    
}
int main(){
    memset(h,-1,sizeof(h));  //初始化对头
    memset(dist,0x3f,sizeof(dist));  //初始化距离,都为最大值
    int n,m;
    cin>>n>>m;
    for(int i=0;i<m;i++){
        int a,b,c;
        cin>>a>>b>>c;
        add(a,b,c);
    }
    spfa();
    if(dist[n]=0x3f3f3f3f) cout<<"impossible";
    else cout<<dist[n];
    return 0;
}