#include<bits/stdc++.h>
using namespace std;

//该题目主要是一个进制数转换的题目

int a;//保存进制数

int main(){
    int n=1000000;
    while(n){
        a+=n%7;
        n/=7;
    }
    cout<<a<<endl;
    return 0;
}