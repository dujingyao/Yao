#include<iostream>
#include<algorithm>
#include<cstring>
using namespace std;
const int N=510;
int Grap[N][N];  //邻接矩阵
int dist[N];  //标记最短距离
int st[N];   //标记是否走过
int pre[N];  //记录前父节点
int n,m;
void prim(){
    memset(dist,0x3f,sizeof(dist));
    int res=0;
    dist[1]=0;
    for(int i=0;i<n;i++){
        int t=-1;
        //t <- 没有连通起来，但是距离连通部分最近的点;
        for(int j=1;j<=n;j++){
            if(!st[j]&&(t==-1||dist[j]<dist[t])){  //如果没有在树中,并且距离树的距离很短
                t=j; //第一轮循环,t=1,dist[t]=0
            }
        }
        if(dist[t]==0x3f3f3f3f){
            cout<<"impossible"<<endl;
            return;
        }
        st[t]=1;
        res+=dist[t];
        for(int j=1;j<=n;j++){
            if(dist[j]>Grap[t][j]&&!st[j]){
                dist[j]=Grap[t][j];
                pre[j]=t;
            }
        }
    }
    cout<<res;
}
void getpath(){   //输出各边
    for(int i=n;i<=n;i++){
        cout<<i<<""<<pre[i]<<endl;
    }
}
int main(){
    memset(Grap,0x3f,sizeof(Grap));
    cin>>n>>m;
    while(m--){
        int a,b,w;
        cin>>a>>b>>w;
        Grap[a][b]=Grap[b][a]=min(Grap[a][b],w);//存储权重最小边
    }
    prim();
    return 0;
}