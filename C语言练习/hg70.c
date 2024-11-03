#include<stdio.h>

int main(){
    double a[11],sum=0.0;
    for(int i=0;i<10;i++){
        scanf("%lf",&a[i]);
        sum+=a[i];
    }
    double max=a[0];
    for(int i=0;i<10;i++){
        if(a[i]>max) max=a[i];
    }
    sum-=max;
    printf("%.2lf",sum/9.0);
    return 0;
}