#include<bits/stdc++.h>
using namespace std;

const int N=(1<<20)+10;
bool st[N];
void get(int n){
    for(int i=2;i<=n;i++){
        if(st[i]) continue;
        for(int i=i+i;j<=n;j+=i){
            st[j]=true;        
        }
    }
}
int n;
int main(){
    get(N);
    while(scanf("%d",&n)!=EOF){
        int k=0,tpl=0;
        while(n>1){
            
        }
    }
    return 0;
}