#include<iostream>
static int size=0;
using namespace std;
int func(int a[],int n);
int main(){
    int N,finalsize;
    cin>>N;
    int a[1000];
    a[0]=1;
    for(int i=1;i<=N;i++){
         finalsize=func(a,i);
    }
    for(int i=finalsize;i>=0;i++){
        cout<<a[i];
    }
    return 0;
}
int func(int a[],int n){
    int f;
    for(int i=size;)
}