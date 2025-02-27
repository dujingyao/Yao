#include<bits/stdc++.h>
using namespace std;

vector<int> a;

int main(){
    
    int n;
    cin>>n;
    for(int i=1;i<=n;i++){
        int x;
        cin>>x;
        a.push_back(x);
    }
    sort(a.begin(),a.begin()+n);
    for(auto it:a){
        cout<<it<<' ';
    }


    return 0;
}