#include<bits/stdc++.h>
using namespace std;
typedef long long ll;

ll l,r,k;
int n,t;

int main(){
    cin>>t;
    while(t--){
        vector<ll> a;
        ll res=0;
        cin>>n>>l>>r>>k;
        for(int i=1;i<=n;i++){
            ll x;
            cin>>x;
            a.push_back(x);
        }
        sort(a.begin(),a.begin()+n);
        int mini=-1,maxi=-1;
        for(int i=0;i<n;i++){
            if(a[i]>=l&&a[i]<=r){
                mini=i;
                break;
            }
        }
        for(int i=n-1;i>=0;i--){
            if(a[i]<=r&&a[i]>=l){
                maxi=i;
                break;
            }
        }
        if(mini==-1){
            cout<<0<<endl;
            continue;
        }
        for(int i=mini;i<=maxi;i++){
            if(k<a[i]) break;
            k-=a[i];
            res++;
        }
        cout<<res<<endl;
    }
    return 0;
}