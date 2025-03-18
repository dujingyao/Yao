#include<bits/stdc++.h>
using namespace std;

bool check(int x){
    int cnt=1;
    while(x){
        if(cnt%2==1&&x%2==0) return false;//奇数位但是该位为偶数
        else if(cnt%2==0&&x%2==1) return false;//偶数位但是该位为奇数
        x/=10;
        cnt++;
    }
    return true;
}

int main(){
    int n,res=0;
    cin>>n;
    for(int i=1;i<=n;i++){
        if(check(i)) res++;
    }
    cout<<res<<endl;
    return 0;
}