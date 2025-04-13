#include<bits/stdc++.h>
using namespace std;

const int N=7510;

int a[N],b[N],st[N];

int n;

int main(){
    
    cin>>n;
    for(int i=1;i<=n;i++) cin>>a[i];
    for(int i=1;i<=n;i++) cin>>b[i];
    int cnt=0;
    for(int i=1;i<=n;i++){
        if(a[i]==b[i]) cnt++;
    }

    for(int i=1;i<=n;i++){
        int l=i,r=i;
        int sum=cnt;
        //以某个点为中心
        while(l>=1&&r<=n){
            sum+=(a[l]==b[r])+(a[r]==b[l])-(a[l]==b[l])-(a[r]==b[r]);
            st[sum]++;
            l--,r++;
        }
        l=i-1,r=i;
        sum=cnt;
        while(l>=1&&r<=n){
            sum+=(a[l]==b[r])+(a[r]==b[l])-(a[l]==b[l])-(a[r]==b[r]);
            st[sum]++;
            l--,r++;
        }
    }
    for(int i=0;i<=n;i++){
        cout<<st[i]<<endl;
    }

    return 0;
}