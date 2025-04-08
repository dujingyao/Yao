#include<bits/stdc++.h>
using namespace std;
char num[20];
int main(){
    cin>>num;
    int a,b;
    //a代表+1
    //b代表-1
    cin>>a>>b;
    for(int i=0;i<strlen(num);i++){
        if(num[i]=='9') continue;
        if(num[i]>='5'&&a>='9'-num[i]){
            a-='9'-num[i];
            num[i]='9';            
        }
        else if(num[i]<='4'&&b>=num[i]-'0'){
            b-=num[i]-'0';
            num[i]='9';
        }
        else if(a>0){
            num[i]+=a;
            a--;
        }
    }
    for(int i=0;i<strlen(num);i++) cout<<num[i];
    return 0;
}