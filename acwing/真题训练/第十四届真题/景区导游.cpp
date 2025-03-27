#include<bits/stdc++.h>
using namespace std;
const int N=1e5+10;
int e[N],ne[N],h[N],w[N],cnt;
void add(int a,int b,int wi){
    e[cnt]=b;
    w[cnt]=wi;
    ne[cnt]=h[a];
    h[a]=cnt++;
}
void init(){
    
}
int main(){
    
    return 0;
}