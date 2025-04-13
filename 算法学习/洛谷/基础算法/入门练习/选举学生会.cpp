#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;
int n,m;
int main(){
    cin>>n>>m;
    vector<int> a(m);
    for(int i=0;i<m;i++){
        cin>>a[i];
    }
    sort(a.begin(),a.end());
    for(int i=0;i<m;i++){
        cout<<a[i]<<" ";
    }

    return 0;
}