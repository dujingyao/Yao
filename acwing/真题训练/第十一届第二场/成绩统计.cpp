#include<bits/stdc++.h>
using namespace std;

int main(){
    int n;
    cin>>n;
    int a=0,b=0;
    for(int i=1;i<=n;i++){
        int x;
        cin>>x;
        if(x>=60) a++;
        if(x>=85) b++;
    }
    cout<<round((double)a/n*100.0)<<'%'<<endl;
    cout<<round((double)b/n*100.0)<<"%"<<endl;
    return 0;
}