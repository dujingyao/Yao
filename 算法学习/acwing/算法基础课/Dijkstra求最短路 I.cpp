#include<iostream>
#include<cstring>
#include<algorithm>
using namespace std;
const int M=1e5+10;//边
const int N=510;  //点
int n,m;
int e[M],ne[M],h[N],w[M],idx;
int dist[N];  //存储最短路径
int s[N];     //存储已确定最短路径的点
int g[N][N];
void add(int a,int b,int c){
    e[idx]=b;
    w[idx]=c;
    ne[idx]=h[a];
    h[a]=idx++;
}
void Dijkstra(){
    memset(dist,0x3f,sizeof(dist));//把所有距离初始化为无穷大
    dist[1]=0;//一号点距离自己的距离是0
    for(int i=0;i<n;i++){
        int t=-1;
        for(int j=1;j<=n;j++){    //确定目前离源点最近的点
            if(!s[j]&&(t==-1||dist[j]<dist[t])){    //没有被标记且是第一个点或者这个点更小
                t=j;
            }
        }
        s[t]=1;   //标记最小点
        for(int j=h[t];j!=-1;j=ne[j]){    //继续向外扩展
            int x=e[j];
            dist[x]=min(dist[x],dist[t]+w[j]);
        }
    }
}
int main(){
    cin>>n>>m;
    memset(h,-1,sizeof(h));
    while(m--){
        int a,b,c;
        cin>>a>>b>>c;
        add(a,b,c);
    }
    Dijkstra();
    if(dist[n]!=0x3f3f3f3f){
        cout<<dist[n]<<endl;
    }
    else cout<<"-1"<<endl;
    return 0;
}