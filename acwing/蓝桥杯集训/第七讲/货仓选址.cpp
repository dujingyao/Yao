#include<bits/stdc++.h>
using namespace std;

const int N=1e5+10;
vector<int> a;
int n;

int main(){
    cin>>n;
    for(int i=1;i<=n;i++){
        int x;
        cin>>x;
        a.push_back(x);
    }
    sort(a.begin(),a.end());
    int res=0;
    int f=a[n/2];
    for(int i=0;i<n;i++){
        res+=abs(a[i]-f);
    }
    cout<<res<<endl;
    return 0;
}