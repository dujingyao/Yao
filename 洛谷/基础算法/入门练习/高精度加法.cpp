#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;
 
vector<int> add(vector<int> a,vector<int> b){
    vector<int> c;
    int t=0;
    for(int i=0;i<a.size()||i<b.size();i++){
        if(i<a.size()) t+=a[i];
        if(i<b.size()) t+=b[i];
        c.push_back(t%10);
        t/=10;
    }
    if(t) c.push_back(1);
    return c;
}
int main(){
    string x,y;
    vector<int> a,b,c;
    cin>>x>>y;

    for(int i=x.size()-1;i>=0;i--){
        a.push_back(x[i]-'0');
    }
    for(int i=y.size()-1;i>=0;i--){
        b.push_back(y[i]-'0');
    }
    c=add(a,b);
    reverse(c.begin(),c.end());
    for(int i=0;i<c.size();i++){
        cout<<c[i];
    }
    return 0;
}