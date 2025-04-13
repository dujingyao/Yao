#include<iostream>
#include<vector>
#include<string>
using namespace std;
vector<int> func(vector<int> a,vector<int> b){
    int t=0,f=1;
    vector<int> c(a.size()+b.size(),0);
    int lena=a.size(),lenb=b.size();
    int len=lena+lenb;
    for(int i=0;i<lena;i++){
        for(int j=0;j<lenb;j++){
            c[i+j]+=a[i]*b[j];
        }
    }
    for(int i=0;i<len;i++){
        c[i+1]+=c[i]/10;
        c[i]%=10;
    }
    while(c.size()>1&&c.back()==0) {
        c.pop_back();
    }
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
    c=func(a,b);
    for(int i=c.size()-1;i>=0;i--){
        cout<<c[i];
    }
    return 0;
}