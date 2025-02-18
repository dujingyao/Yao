#include<iostream>
using namespace std;
int n;
int st[20];//标记  0:不确定 1：选 2：不选
void dfs(int u){
    if(u>n){
        for(int i=1;i<=n;i++){
            if(st[i]==1){
                cout<<i<<' ';
            }
        }
        cout<<endl;
        return;
    }

    st[u]=2;
    dfs(u+1);
    st[u]=0;
    st[u]=1;
    dfs(u+1);
    st[u]=0;
}
int main(){
    cin>>n;
    dfs(1);
    return 0;
}