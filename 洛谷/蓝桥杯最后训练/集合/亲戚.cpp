#include<bits/stdc++.h>
using namespace std;
const int N=5010;
int fa[N];
int n,m,p,x,y;
int find(int d){
    int c;
    if(d==fa[d]) return d;
    for(int i=d;;i=fa[i]){
        if(i==fa[i]) return i;
    }
}
void join(int c1,int c2){
    int f1=find(c1),f2=find(c2);
    if(f1!=f2) fa[f1]=f2;
}
int main(){
    cin>>n>>m>>p;
    for(int i=1;i<=n;i++) fa[i]=i;
    for(int i=1;i<=m;i++){
        cin>>x>>y;
        join(x,y);
    }
    for(int i=1;i<=p;i++){
        cin>>x>>y;
        if(find(x)==find(y)) cout<<"Yes"<<endl;
        else cout<<"No"<<endl;
    }
    return 0;
}