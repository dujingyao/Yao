#include<bits/stdc++.h>
using namespace std;
#define esp 1e-4
double a,b,c,d;
double f(double x){
    return a*x*x*x+b*x*x+c*x+d;
}

int main(){
    cin>>a>>b>>c>>d;
    for(int i=-100;i<=100;i++){
        double l=i,r=i+1,mid;
        if(fabs(f(l))<esp) printf("%.2lf ",l);
        else if(fabs(f(r))<esp) continue;
        else if(f(l)*f(r)<0){//一定有一个值
            while(r-l>esp){
                mid=(l+r)/2;
                if(f(mid)*f(r)>0) r=mid;
                else l=mid; 
            }
            printf("%.2lf ",l);
        }
    }
    return 0;
}