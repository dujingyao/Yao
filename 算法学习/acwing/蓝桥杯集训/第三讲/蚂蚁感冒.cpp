#include<bits/stdc++.h>
using namespace std;

const int N=60;
int a[N];

int main(){
    
    int n;
    cin>>n;
    int l=0,r=0;
    cin>>a[1];
    int x=abs(a[1]);
    for(int i=2;i<=n;i++){
        cin>>a[i];
        if(a[1]>0){//向右
            if(abs(a[i])>x&&a[i]<0) l++;
            if(abs(a[i])<x&&a[i]>0) r++;
        }
        if(a[1]<0){
            if(abs(a[i])<x&&a[i]>0) r++;
            if(abs(a[i])>x&&a[i]<0) l++;
        }
    }
    if(a[1]>0&&l==0||a[1]<0&&r==0){
        cout<<1<<endl;
    }
    else{
        cout<<l+r+1<<endl;
    }

    return 0;
}