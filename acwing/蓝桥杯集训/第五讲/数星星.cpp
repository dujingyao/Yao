#include<bits/stdc++.h>
using namespace std;

const int N=32010;

int a[N],tr[N];

int n;

int lowbit(int x){
    return x&-x;
}

int add(int x){
    for(int i=x;i<N;i+=lowbit(i)) tr[i]++;
}

int query(int x){
    int res=0;
    for(int i=x;i;i-=lowbit(i)) res+=tr[i];
    return res;
}

int main(){
    
    cin>>n;
    for(int i=0;i<n;i++){
        int x,y;
        cin>>x>>y;
        x++;
        a[query(x)]++;
        add(x);
    }
    for(int i=0;i<n;i++){
        cout<<a[i]<<endl;
    }
    return 0;
}