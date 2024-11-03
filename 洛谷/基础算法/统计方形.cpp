#include<iostream>
using namespace std;
int main(){
    long long x=0,y=0,n,m;
    cin>>n>>m;
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            if(i==j) x+=(n-i)*(m-j);
            else y+=(n-i)*(m-j);
        }
    }
    cout<<x<<" "<<y<<endl;
    return 0;
}