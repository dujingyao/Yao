#include<stdio.h>

int main(){
    double a,b,c;
    scanf("%lf %lf %lf",&a,&b,&c);
    printf("%.2f",(a+b)*c/2.0);
    return 0;
}