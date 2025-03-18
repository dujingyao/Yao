#include<bits/stdc++.h>
using namespace std;

typedef pair<double,double> PII;

const int N=1010;
PII g[N];

bool cmp(PII a,PII b){
    return a.second<b.second;
}

double x,y,l,r;
int n,d,res=1;

int main(){
    cin>>n>>d;
    for(int i=1;i<=n;i++){
        cin>>x>>y;
        if(y>d){
            cout<<-1<<endl; 
            return 0;
        }
        double offset=sqrt(d*d-y*y);
        l=x-offset;
        r=x+offset;
        g[i]={l,r};//把每个点的范围加入进去
    }
    //根据右端点从小到大排序
    sort(g+1,g+n+1,cmp);
    double ri=g[1].second;
    for(int i=2;i<=n;i++){
        if(g[i].first<=ri) continue;
        res++;
        ri=g[i].second;
    }
    cout<<res<<endl;
    return 0;
}