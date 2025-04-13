#include<bits/stdc++.h>
using namespace std;
const int N=100010;
int n,m,res,ans;
int school[N],student[N];
int main(){
    cin>>n>>m;
    for(int i=1;i<=n;i++) cin>>school[i];
    for(int i=1;i<=m;i++) cin>>student[i];
    sort(school+1,school+1+n);
    sort(student+1,student+1+m);
    for(int i=1;i<=m;i++){
        int l=1,r=n,mid;
        int minx,maxx;
        //找最后一个小于学生分数的学校
        while(l<r){
            mid=(l+r)/2;
            if(school[mid]>=student[i]) r=mid;
            else l=mid+1;
        }
        minx=l;
        l=1,r=n;
        while(l<r){
            mid=(l+r+1)/2;
            if(school[mid]<=student[i]) l=mid;
            else r=mid-1;
        }
        maxx=l;
        res=min(abs(school[maxx]-student[i]),abs(school[minx]-student[i]));
        ans+=res;
    }
    cout<<ans;
    return 0;
}