#include<iostream>
#include<algorithm>
using namespace std;

const int N=110,V=110;
int s[N],v[N][N],w[N][N];
int n,m;
int f[N];
int main(){
    cin>>n>>m;
    for(int i=1;i<=n;i++){
        cin>>s[i];
        for(int j=1;j<=s[i];j++){
            cin>>v[i][j]>>w[i][j];
        }
    }

    for(int i=1;i<=n;i++){  //组别
        for(int j=m;j>=0;j--){  //体积
            for(int k=0;k<=s[i];k++){  //每组物品的遍历
                if(v[i][k]<=j){
                    f[j]=max(f[j],f[j-v[i][k]]+w[i][k]);
                }
            }
        }
    }
    cout<<f[m]<<endl;
    return 0;
}