#include<iostream>
#include<algorithm>
#define MAX 9
using namespace std;
int n;
int a[10];
int b[MAX];
void dfs(int u){
    if(u>n){
        for(int i=1;i<=n;i++){
            cout<<b[i]<<" ";
        }
        cout<<endl;
        return;
    }
    for(int i=1;i<=n;i++){
        if(!a[i]){
            a[i]=1;
            b[u]=i;
            dfs(u+1);
            a[i]=0;
        }
    }
}
int main(){
    cin>>n;
    dfs(1);
    return 0;
}