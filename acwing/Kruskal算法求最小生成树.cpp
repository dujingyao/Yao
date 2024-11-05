#include<iostream>
#include<algorithm>
#include<cstring>
using namespace std;
const int N=100010;
int p[N];  //保存并查集

struct E{
    int a;
    int b;
    int w;
    bool operator <(const E& rhd){//通过边长进行排序
        return this->w<rhd.w;
    }
}edg[N*2];
int res=0;
int n,m;
int cnt=0;
int find(int a){   //找祖宗,主要用来判断是否有同一个根结点
    if(p[a]!=a) p[a]=find(p[a]);
    return p[a];
}
void Kruskal(){
    for(int i=1;i<=m;i++){
        int pa=find(edg[i].a);
        int pb=find(edg[i].b);
        if(pa!=pb){  //如果不是同一个祖宗,也就是不在同一个集合当中
            res+=edg[i].w;
            p[pa]=pb;
            cnt++;  //保留的边数加一
        }
    }
}

int main(){
    cin>>n>>m;
    for(int i=1;i<=n;i++){
        p[i]=i;
    }
    for(int i=1;i<=m;i++){
        int a,b,c;
        cin>>a>>b>>c;
        edg[i]={a,b,c};
    }
    sort(edg+1,edg+m+1);
    Kruskal();
    if(cnt<n-1) cout<<"impossible";
    else cout<<res;
    return 0;
}