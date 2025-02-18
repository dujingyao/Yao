#include<iostream>
using namespace std;

int n;

int a[10];
bool st[10];//标记

void dfs(int u){
    if(u>n){
        for(int i=1;i<=n;i++){
            cout<<a[i]<<' ';
        }
        cout<<endl;
        return;
    }
   for(int i=1;i<=n;i++){
        if(!st[i]){
            st[i]=true;
            a[u]=i;
            dfs(u+1);
            a[u]=0;
            st[i]=false;
        }
   }
}

int main(){
    cin>>n;
    dfs(1);
    return 0;
}