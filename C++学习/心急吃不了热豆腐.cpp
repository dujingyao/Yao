#include<iostream>
#include<string>
using namespace std;

int main(){
    int n;
    cin>>n;
    string ch;
    cin>>ch;
    for(int i=0;i<ch.size();i++){
        if(ch[i]>='A'&&ch[i]<='Z') ch[i]+=32;
        else ch[i]-=32;
    }
    cout<<ch<<endl;
    return 0;
}