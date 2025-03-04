#include<bits/stdc++.h>
using namespace std;

const int N=40;

int s[N][N];

int main(){
    int n,m;
    cin>>n>>m;
    if(n%2==0&&m%2==0){
        cout<<0;
        return 0;
    }
    s[1][1]=1;
    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++){
            if(i==1&&j==1) continue;
            if(i%2||j%2){
                s[i][j]=s[i-1][j]+s[i][j-1];
            }
        }
    }
    cout<<s[n][m];
    return 0;
}