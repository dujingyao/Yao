#include<bits/stdc++.h>
using namespace std;

int main(){
    
    int n;
    cin>>n;
    int res=0;
    for(int i=0;i<=n;i++){
        int x=i;
        while(x){
            int y=x%10;
            if(y==0||y==1||y==2||y==9){
                res+=i;
                break;
            }
            x/=10;
        }
    }
    cout<<res<<endl;
    return 0;
}