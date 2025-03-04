#include<bits/stdc++.h>
using namespace std;

int main(){
    string st;
    cin>>st;
    int len=st.length();
    for(int i=0;i<len;i++){
        if(st[i]>'9'||st[i]<'0') cout<<st[i];
        if(st[i]<='9'&&st[i]>='0'){
            for(int j=1;j<=st[i]-'0'-1;j++){
                cout<<st[i-1];
            }
        }

    }
    return 0;
}