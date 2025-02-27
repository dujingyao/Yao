#include<iostream>
#include<cstring>
#include<queue>
#include<algorithm>
using namespace std;
const int N=1e5+10;
int h[N],e[N],ne[N],idx;
int d[N];//标记
int q[N],hh=0,tt=-1;//用数组模拟队列,hh是头指针,tt是尾指针
int n,m;
int t[N];
void add(int a,int b){
    e[idx]=b;
    ne[idx]=h[a];
    h[a]=idx++;
    d[b]++;
}
//若入度为0,则入队
bool topsort(){
    //先遍历结点看入度是否为0,若为0则入队
    for(int i=1;i<=n;i++){
        if(!d[i]){
            q[++tt]=i;//入队,尾指针加一
        }
    }
    while(hh<=tt){//队列非空
        int a=q[hh++];
        for(int i=h[a];i!=-1;i=ne[i]){
            int b=e[i];
            d[b]--;//已经访问过了,删除
            if(!d[b]){//入度为0则入队
                q[++tt]=b;
            }
        }
    }
    return tt==n-1;
}
int main(){
    memset(h,-1,sizeof(h));
    memset(d,0,sizeof(d));//入度初始化为0
    cin>>n>>m;
    for(int i=1;i<=m;i++){
        int a,b;
        cin>>a>>b;
        add(a,b);
    }
    if(topsort()){
        for(int i=0;i<n;i++){
            printf("%d ",q[i]);
        }
    }
    else cout<<"-1"<<endl;
    return 0;
}