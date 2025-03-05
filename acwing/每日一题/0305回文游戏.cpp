#include<bits/stdc++.h>
using namespace std;
int main(){
    int T;
    cin>>T;
    while(T--){
        string ch;
        cin>>ch;
        if(ch[ch.size()-1]=='0'){
            cout<<'E'<<endl;
        }
        else cout<<'B'<<endl;

    }
    return 0;
}