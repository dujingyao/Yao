#include<bits/stdc++.h>
using namespace std;

int max1=0,min1=100;

double sum=0;

int main(){
    
    int n;
    cin>>n;
    for(int i=1;i<=n;i++){
        int grade;
        cin>>grade;
        max1=max(max1,grade);
        min1=min(min1,grade);
        sum+=grade;
    }
    printf("%d\n%d\n%.2lf",max1,min1,sum/(1.0*n));

    return 0;
}