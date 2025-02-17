#include<iostream>
using namespace std;
int N,Q,x,y,z;
int a[1010][1010],b[1010][1010],c[1010][1010];
int main(){
    
    cin>>N>>Q;
    int res=0;
    while(Q--){
        cin>>x>>y>>z;
        a[x][y]++;
        b[x][z]++;
        c[y][z]++;
        if(a[x][y]>=N) res++;
        if(b[x][z]>=N) res++;
        if(c[y][z]>=N) res++;
        printf("%d\n",res);
    }

    return 0;
}