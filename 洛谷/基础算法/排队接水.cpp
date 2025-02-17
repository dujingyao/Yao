#include<iostream>
#include<algorithm>
using namespace std;

struct water{
    int num,time;
}p[1010];

bool cmp(water a,water b){
    if(a.time!=b.time){
        return a.time<b.time;
    }     
}

int main(){
    int n;
    cin>>n;
    for(int i=1;i<=n;i++){
        cin>>p[i].time;
        p[i].num=i;
    }
    sort(p+1,p+n+1,cmp);
    long long sum=0;
    for(int i=1;i<=n;i++){
        cout<<p[i].num<<' ';
        sum+=i*p[n-i].time;
    }
    printf("\n%.2lf\n",1.0*sum/n);

    return 0;
}