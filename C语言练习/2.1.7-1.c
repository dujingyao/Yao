#include<stdio.h>
#include<string.h>
// const int N=1e6+10;
int n;
int path[(int)1e6+10],st[(int)1e6+10];

void dfs(int u){
    if(u==n+1){
        for(int i=1;i<=n;i++)
            printf("%d",path[i]);
        putchar('\n');
        return ;
    }
    for(int i=1;i<=n;i++){
        if(!st[i]){
            st[i]=1;
            path[u]=i;
            dfs(u+1);
            st[i]=0;
        }
    }
}

void solve(){
    memset(st,0,sizeof st);
    scanf("%d",&n);
    dfs(1);
}

int main(){
    int T=1;
    scanf("%d",&T);
    while(T--) solve();
    return 0;
}