#include<bits/stdc++.h>
using namespace std;

int main(){
    
    int n;
    cin>>n;
    int sum=n;
    if(n<3){
        cout<<n<<endl;
        return 0;
    }
    while(n>=3){
        sum+=1;
        if(n>=3){
            n-=3;
            n+=1;
        }
    }
    cout<<sum<<endl;
    return 0;
}