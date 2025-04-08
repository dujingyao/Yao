#include<bits/stdc++.h>
using namespace std;
const int N=1e7;
bool check(int u){
    int i=1;
    while(u){
        if(i%2==1&&u%10%2==1){
            i++;
            u/=10;
            continue;
        }
        else if(i%2==0&&u%10%2==0){
            i++;
            u/=10;
            continue;
        }
        else{
            return false;
        }
    }
    return true;
}
int main(){
    int n,ans=0;
    cin>>n;

    for(int i=1;i<=n;i++){
        if(check(i)) ans++;
    }
    cout<<ans;
    return 0;
}