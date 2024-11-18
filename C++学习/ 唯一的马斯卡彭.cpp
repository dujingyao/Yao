#include<iostream>
#include<algorithm>
#include<vector>
using namespace std;

int main(){
    int n,m;
    cin>>m;
    while(m--){
        vector<int> a;
        cin>>n;
        while(n--){
            int x;
            cin>>x;
            a.push_back(x);  
        }
        sort(a.begin(),a.end());
        while(a.size()>1){
            a.erase(a.begin());
            if(a.size()!=1) a.pop_back();
        }
        cout<<a[0]<<endl;
    }
    return 0;
}