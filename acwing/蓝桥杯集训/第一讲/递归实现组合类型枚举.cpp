#include<iostream>
using namespace std;

int n,m;
int st[25];

void dfs(int u,int start){
    if(u==m+1){
        for(int i=1;i<=m;i++){
            cout<<st[i]<<' ';
        }
        cout<<endl;
        return;
    }
    for(int i=start;i<=n;i++){
        st[u]=i;
        dfs(u+1,i+1);
        st[u]=0;
    }
}

int main(){
    cin>>n>>m;
    
    dfs(1,1);

    return 0;
}