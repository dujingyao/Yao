#include<iostream>
#include<cstring>
#include<algorithm>
using namespace std;
const int N=510;  //点
const int M=1e5+10;  //边
int m,n,k;
int dist[N];   //记录距离
int last[N];   //备份数据防止串联
struct e{
    int a;
    int b;
    int w;
}e[M];
void bellman_ford(){
    memset(dist,0x3f,sizeof(dist));
    dist[1]=0;
    for(int i=0;i<k;i++){
        memcpy(last,dist,sizeof(dist));  //拷贝
        for(int j=0;j<m;j++){
            auto h=e[j];
            dist[h.b]=min(dist[h.b],last[h.a]+h.w);
        }
    }
}
int main(){
    cin>>n>>m>>k;
    for(int i=0;i<m;i++){
        int a,b,c;
        cin>>a>>b>>c;
        e[i]={a,b,c};
    }
    bellman_ford();
    if(dist[n]>0x3f3f3f3f/2) puts("impossible");
    else cout<<dist[n]<<endl;
    return 0;
}