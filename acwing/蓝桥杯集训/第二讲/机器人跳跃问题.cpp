#include<iostream>
#include<cmath>
using namespace std;

const int N=1e5+10;

int h[N];
double n,x=2;
double e=0;
int main(){
    cin>>n;
    for(int i=1;i<=n;i++){
        cin>>h[i];
        e=e+1.0*h[i]/x;
        x*=2;
    }
    cout<<ceil(e)<<endl;
    return 0;
}