#include<iostream>
#include<algorithm>
using namespace std;
struct compate{
    int l,r;
}p[100010];
bool cmp(compate a,compate b){
    return a.r<=b.r;
}
int n;

int main(){
    cin>>n;
    for(int i=1;i<=n;i++){
        cin>>p[i].l>>p[i].r;
    }
    sort(p+1,p+n+1,cmp);
    int res=0,max=0;
    for(int i=1;i<=n;i++){
        if(p[i].l>=max){
            res++;
            max=p[i].r;
        }
    }
    cout<<res<<endl;
    return 0;
}