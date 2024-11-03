#include<iostream>
#include <algorithm>
#include <cstring>
using namespace std;
const int N=1e5+10;   //最大结点数
const int M=N*2;      //最大边数
int e[M],ne[M],h[N],idx;
bool st[N];   //标记这个点是否被遍历
int ans=N;
int n;
//a为根节点,把b加到a后
void add(int a,int b){
    e[idx]=b;
    ne[idx]=h[a];
    h[a]=idx++;
}
int dfs(int u){
    st[u]=true;
    int res=0;
    int sum=1;//以u为根的结点数
    for(int i=h[u];i!=-1;i=ne[i]){
        int j=e[i];
        if(!st[j]){//如果这个点没有被访问过
            int s=dfs(j);
            res=max(res,s);
            sum+=s;
        }
    }
    res=max(res,n-sum);//某个点的最大的联通图
    ans=min(res,ans);//所有点的最大联通图的最小数
    return sum;
}
int main(){
    cin>>n;
    memset(h,-1,sizeof h);
    for(int i=0;i<n-1;i++){
        int a,b;
        cin>>a>>b;
        add(a,b);
        add(b,a);
    }
    dfs(1);
    cout<<ans<<endl;
    return 0;
}