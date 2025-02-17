#include<iostream>
using namespace std;

const int N=1010;
int a[N];
int f[N];  //所有以n结尾的最大上升子序列
int n;

int main(){
    
    cin>>n;
    for(int i=1;i<=n;i++){
        cin>>a[i];
    }

    for(int i=1;i<=n;i++){
        f[i]=1;
        for(int j=1;j<i;j++){
            if(a[i]>a[j]){
                f[i]=max(f[j]+1,f[i]);
            }
        }
    }

    int res=f[1];
    for(int i=1;i<=n;i++){
        if(res<=f[i]) res=f[i];
    }

    cout<<res;

    return 0;
}