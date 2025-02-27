#include<iostream>
#include<algorithm>
#include<cstring>
using namespace std;
const int N=100010*2;
int h[N],e[N],ne[N],idx;
int n,m;
//未染色为0,红色为1,黑色为2
//如果已经染色,假如颜色为c,可以用3-c来判断
int color[N];   
void add(int a,int b){
    e[idx]=b;
    ne[idx]=h[a];
    h[a]=idx++;
}
bool dfs(int u,int c){
    color[u]=c;
    for(int i=h[u];i!=-1;i=ne[i]){
        int b=e[i];
        if(!color[b]){  //若未染过色
            if(!dfs(b,3-c)) return false;
        }
        else if(color[b]&&color[b]!=3-c){  //如果已经染色并且颜色一致
            return false;
        }
    }
    return true;
}
int main(){
    memset(h,-1,sizeof(h));
    cin>>n>>m;
    while(m--){
        int a,b;
        cin>>a>>b;
        add(a,b),add(b,a);
    }
    for(int i=1;i<=n;i++){
        if(!color[i]){  //如果有没染过色的
            if(!dfs(i,1)){
                cout<<"No"<<endl;
                return 0;
            }
        }
    }
    cout<<"Yes"<<endl;

    return 0;
}