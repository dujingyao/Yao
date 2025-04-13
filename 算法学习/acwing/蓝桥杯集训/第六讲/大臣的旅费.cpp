#include<bits/stdc++.h>
using namespace std;

//求树的直径
//1.任取一个点x
//2.找到距离x最远的点y
//3.再找一个离x最远的点，即为答案
const int N=1e5+10;
struct road{
    int id,w;//id表示位置，w表示长度
};
int n,dis[N];//存每个点到u点的距离
vector<road> t[N];
void dfs(int u,int father,int distance){
    dis[u]=distance;
    for(auto node:t[u]){//遍历自己的子节点
        if(node.id!=father){//避免回到父节点
            dfs(node.id,u,distance+node.w);
        }
    }
}
int main(){
    cin>>n;
    for(int i=1;i<=n;i++){
        int x,y,w;
        cin>>x>>y>>w;
        t[x].push_back({y,w});
        t[y].push_back({x,w});
    }
    dfs(1,-1,0);
    int u=1;
    for(int i=1;i<=n;i++){
        if(dis[i]>dis[u]) u=i;
    }
    dfs(u,-1,0);
    for(int i=1;i<=n;i++){
        if(dis[i]>dis[u]) u=i;
    }
    int s=dis[u];
    cout<<s*10+s*(s+1ll)/2<<endl;
    return 0;
}