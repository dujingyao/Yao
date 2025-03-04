#include<iostream>
#include<algorithm>
#define size 5
using namespace std;
int a[size*size],n=4*4,ans=0;
int b1[size][5],b2[size][5],b3[size][5];    //分别记录行,列,四小格
void dfs(int t){
    if(t>n){
        ans++;
        for(int i=1;i<=n;i++){
            cout<<a[i]<<" ";
            if(i%4==0) cout<<endl;
        }
        cout<<endl;
        return;
    }
    int row=(t-1)/4+1;
    int col=(t-1)%4+1;
    int block=(row-1)/2*2+(col-1)/2+1;
    for(int i=1;i<=4;i++){
        if(b1[row][i]==0&&b2[col][i]==0&&b3[block][i]==0){
            a[t]=i;
            b1[row][i]=1;b2[col][i]=1;b3[block][i]=1;
            dfs(t+1);
            b1[row][i]=0;b2[col][i]=0;b3[block][i]=0;
        }
    }
}

int main(){
    dfs(1);
    cout<<ans<<endl;
    return 0;
}