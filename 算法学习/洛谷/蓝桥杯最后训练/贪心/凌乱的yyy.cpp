#include<bits/stdc++.h>
using namespace std;

const int N=1e5+10;
struct race{
    int start,end;
}r[N];
bool cmp(race a,race b){
    return a.end<b.end;
}
int n;
int main(){
    scanf("%d",&n);
    for(int i=1;i<=n;i++){
        scanf("%d%d",&r[i].start,&r[i].end);
    }
    sort(r+1,r+1+n,cmp);
    int ans=1,lasttime=r[1].end;
    for(int i=2;i<=n;i++){
        if(r[i].start>=lasttime){
            ans++;
            lasttime=r[i].end;
        }
    }
    printf("%d",ans);

    return 0;
}