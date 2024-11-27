#include<iostream>
#include<algorithm>
#include<cmath>
using namespace std;

int main(){
    int n;
    cin>>n;
    int a[n+2][n+2];
    for(int i=0;i<n+2;i++){
        for(int j=0;j<n+2;j++){
            a[i][j]=0;
        }
    }
    for(int i=1;i<=n;i++){
        for(int j=1;j<=i;j++){
            cin>>a[i][j];
        }
    }
    int max1=0;
    if(n>1){
        for(int i=2;i<=n;i++){
            for(int j=1;j<=i;j++){
                a[i][j]+=max(a[i-1][j-1],a[i-1][j]);
                if(max1<a[i][j]) max1=a[i][j];
            }
        }
    }else max1=a[1][1];
    cout<<max1<<endl;
    return 0;
}