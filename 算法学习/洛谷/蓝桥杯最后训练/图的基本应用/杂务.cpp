#include<bits/stdc++.h>
using namespace std;
const int N=1e4+10;
int n,x,y,t,ans,len[N],vis[N];
vector<int> linker[N];
int dfs(int x){
    //如果当前节点以及访问，直接返回记忆化的结果
    if(vis[x]) return vis[x];
    int maxti=0;//记录前置任务的最大时间
    for(int i=0;i<linker[x].size();i++){//遍历前置节点的最大值
        maxti=max(maxti,dfs(linker[x][i]));
    }
    vis[x]=maxti+len[x];
    return vis[x];
}
int main(){
    cin>>n;
    for(int i=1;i<=n;i++){
        cin>>x>>len[i];//当前节点以及长度
        while(cin>>y){//输入需要前置完成的节点
            if(!y) break;
            else linker[y].push_back(x);
        }
    }    
    for(int i=1;i<=n;i++){
        ans=max(ans,dfs(i));
    }
    cout<<ans<<endl;
    return 0;
}