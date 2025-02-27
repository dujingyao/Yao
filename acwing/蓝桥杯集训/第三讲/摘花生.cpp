#include<bits/stdc++.h>
using namespace std;

const int N=1010;
int a[N][N];
int t;
int f[N][N];

int main(){

    cin>>t;
    while(t--){
        int x,y;
        cin>>x>>y;
        for(int i=1;i<=x;i++){
            for(int j=1;j<=y;j++){
                cin>>a[i][j];
            }
        }
        for(int i=1;i<=x;i++){
            for(int j=1;j<=y;j++){
                f[i][j]=max(f[i-1][j],f[i][j-1])+a[i][j];
            }
        }
        cout<<f[x][y]<<endl;
    }


    return 0;
}