#include<stdio.h>

int main(){
    double sum=0,n;
    int i;
    for(i=1;i<=1000;i++){
        scanf("%lf",&n);
        sum+=n;
        if(sum>=20000) break;
    }
    printf("%d %.2lf",i,sum/i);
    return 0;
}