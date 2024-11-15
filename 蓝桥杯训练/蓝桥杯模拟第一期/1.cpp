#include<iostream>
#include<algorithm>
using namespace std;
int get(int x){
    for(int j=2;j<=x/j;j++){
        if(x%j==0) return 0;
    }
    return 1;
}
int main(){
    int res=0;
    for(int i=2;i<=2024;i++){
        if(2024%i==0&&get(i)){
            res++;
            cout<<i<<' ';
        }
    }
    cout<<res<<endl;
    return 0;
}