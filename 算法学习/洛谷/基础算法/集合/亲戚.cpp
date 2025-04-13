#include<bits/stdc++.h>
using namespace std;

const int N=5010;

int x,y;
int n,m,p;

int fa[N];


//检查
int check(int a){
    if(a==fa[a]) return a;
    return fa[a]=check(fa[a]); 
}

//加入
void join(int c1,int c2){
    //看看家里长辈都是谁
    int f1=check(c1),f2=check(c2);
    if(f1!=f2) fa[f1]=f2; 
}

int main(){
    
    cin>>n>>m>>p;
    for(int i=1;i<=n;i++){
        fa[i]=i;
    }
    for(int i=1;i<=m;i++){
        int x,y;
        cin>>x>>y;
        join(x,y);
    }
    for(int i=1;i<=p;i++){
        int x,y;
        cin>>x>>y;
        if(check(x)==check(y)) cout<<"Yes"<<endl;
        else cout<<"No"<<endl;
    }

    return 0;
}