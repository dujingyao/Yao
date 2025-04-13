#include<bits/stdc++.h>
using namespace std;

const int N=10010;
int a[N];
bool st[N];//判断是否已经被加入
int n;

int main(){
    cin>>n;
    int res=0;
    for(int i=1;i<=n;i++) cin>>a[i];
    for(int i=1;i<=n;i++){
        if(!st[i]){
            res++;
            for(int j=i;;j=a[j]){
                if(st[j]==true) break;
                st[j]=true;
            }
        }
    }
    cout<<n-res<<endl;
    return 0;
}