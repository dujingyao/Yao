#include<iostream>
#include<vector>
#include<string>
using namespace std;
vector<int> cmp(vector<int>& a,vector<int>& b){
    vector<int> c;
    for(int i=0,t=0;i<a.size();i++){
        t=a[i]-t;
        if(i<b.size()){
            t-=b[i];
        }
        c.push_back((t+10)%10);
        if(t<0) t=1;
        else t=0;
    }
    return c;
}
int main(){
    string A;
    cin>>A;
    int f=0;
    vector<int> a,b,c;
    for(int i=A.size()-1;i>=0;i--){
        if(f==0){
            if(A[i]!='-'){
                b.push_back(A[i]-'0');
            }else{
                f=1;
                continue;
            }
        }
        if(f==1){
            a.push_back(A[i]-'0');
        }
    }
    c=cmp(a,b);
    for(int i=c.size()-1;i>=0;i--){
        cout<<c[i];
    }
    return 0;
}