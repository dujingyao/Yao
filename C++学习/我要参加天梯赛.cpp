#include<iostream>
using namespace std;

const int N=1e9+7;

int main(){
    
    int n;
    cin>>n;
    int res=0,goal=0;
    for(int i=0;i<=n;i++){
        for(int j=0;2*j+i<=n;j++){
            for(int k=0;i+2*j+3*k<=n;k++){
                if(i+2*j+3*k==n){
                    cout<<i<<' '<<j<<' '<<k<<endl;
                    res++;
                    res%=N;
                }
            }
        }
    }
    cout<<res<<endl;
    return 0;
}