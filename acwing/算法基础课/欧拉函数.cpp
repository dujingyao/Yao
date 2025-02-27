#include<iostream>
#include<algorithm>
using namespace std;

void phi(int x){
    int res=x;
    for(int i=2;i<=x/i;i++){
        if(x%i==0){
            res=res/i*(i-1);//公式
            while(x%i==0) x/=i;
        }
    }
    if(x>1) res=res/x*(x-1);
    cout<<res<<endl;
}

int main(){
    int n;
    cin>>n;
    while(n--){
        int x;
        cin>>x;
        phi(x);
    }
    return 0;
}