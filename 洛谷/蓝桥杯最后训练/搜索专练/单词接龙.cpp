#include<bits/stdc++.h>
using namespace std;

const int N=30;
string ch[2*N];
char start;
int n;

int main(){
    cin>>n;
    for(int i=1;i<=n-1;i++){
        cin>>ch[i];
        ch[i+n];//复制一份
    }  
    cin>>start;
    dfs(start);

    return 0;
}