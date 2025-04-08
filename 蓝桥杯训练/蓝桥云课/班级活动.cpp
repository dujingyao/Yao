#include<bits/stdc++.h>
using namespace std;
const int N=1e5+10;
int a[N],b[N];
int n,sum;
int main(){
    cin>>n;
    for(int i=1;i<=n;i++){
        cin>>a[i];
        b[a[i]]++;
    }
    int x=0,y=0;//奇偶
    for(int i=1;i<=1e5;i++){
        if(!b[i]) continue;
        if(b[i]==1) x++;
        else if(b[i]>2) y+=(b[i]-2);
    }
   if(y>x) sum=y;
   else sum=(x-y)/2+y;
   cout<<sum;
    return 0;
}