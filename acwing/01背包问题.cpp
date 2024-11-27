#include<iostream>
#include<algorithm>
using namespace std;

const int N=1010;

int n,m;
int V[N],W[N];

int f[N][N];

int main(){
    
    cin>>n>>m;
    for(int i=1;i<=n;i++){
        cin>>V[i]>>W[i];
    }
    for(int i=0;i<=m;i++){
        f[0][i]=0;
    }
    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++){
            if(V[i]<=j){
                f[i][j]=max(f[i-1][j],f[i-1][j-V[i]]+W[i]);
            }
            else{
                f[i][j]=f[i-1][j];
            }
        }
    }
    cout<<f[n][m]<<endl;

    return 0;
}