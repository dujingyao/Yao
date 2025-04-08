#include<bits/stdc++.h>
using namespace std;

const int N=1010;
struct times{
    double x;
    int y;
}ti[N];
bool cmp(times a,times b){
    return a.x<b.x;
}
int n;
double s[N];
int main(){
    cin>>n;
    for(int i=1;i<=n;i++){
        cin>>ti[i].x;
        ti[i].y=i;
    }
    sort(ti+1,ti+n+1,cmp);
    double sum=0;
    for(int i=1;i<=n;i++){
        cout<<ti[i].y<<' ';
        sum+=i*ti[n-i].x;
    }
    cout<<endl;
    printf("%.2lf",sum/(1.0*n));
    return 0;
}