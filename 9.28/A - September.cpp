#include<iostream>
#include<string>
#include<string.h>
using namespace std;

int main(){
    string ch;
    int i;
    for(i=1;i<=12;i++){
        cin>>ch;
        if(i==ch.length()) cout<<ch<<endl;
        i++;
    }
    return 0;
}