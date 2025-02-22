#include<iostream>
using namespace std;

const int N=1e6+10;

int num[10];//记录九个数的全排列
bool st[10];//记录是否被标记

int n,cnt=0;

int cala(int l,int r){
    int res=0;
    for(int i=l;i<=r;i++){
        res=res*10+num[i];
    }
    return res;
}

//生成九个数的全排列
void dfs(int u){
    if(u==9){//已经排列好
        for(int i=0;i<7;i++){
            for(int j=i+1;j<8;j++){
                int a=cala(0,i);
                int b=cala(i+1,j);
                int c=cala(j+1,8);
                if(a*c+b==c*n){
                    cnt++;
                }
            }

        }
        return;
    }
    //全排列
    for(int i=1;i<=9;i++){
        if(!st[i]){
            st[i]=true;
            num[u]=i;
            dfs(u+1);
            st[i]=false;
        }
    }
}

int main(){
    cin>>n;
    dfs(0);
    cout<<cnt;

    return 0;
}