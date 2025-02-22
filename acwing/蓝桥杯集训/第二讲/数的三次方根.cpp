#include<iostream>
using namespace std;

int main(){
    double l=-10000,r=10000;
    double x;
    cin>>x;
    double mid;
    while(r-l>1e-8){
        mid=(l+r)/2;
        if(mid*mid*mid>=x) r=mid;
        else l=mid;
    }
    printf("%.6lf",l);
    return 0;
}