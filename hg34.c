#include<stdio.h>

int main(){
    int a[11];
    for(int i=0;i<10;i++){
        scanf("%d",&a[i]);
    }
    int max=a[0],min=a[0];
    double sum=0;
    for(int i=0;i<10;i++){
        if(a[i]>max) max=a[i];
        if(a[i]<min) min=a[i];
        sum+=a[i];
    }
    sum=sum-max-min;
    double x;
    x=1.0*sum/8;
    printf("%.2f",x);
    return 0;
}