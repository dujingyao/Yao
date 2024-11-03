#include<stdio.h>
#include<math.h>
int main(){
    int a,b,c;
    scanf("%d %d %d",&a,&b,&c);
    double x1,x2;
    x1=(-b+sqrt(b*b-4*a*c))/(2*a)*1.0,x2=1.0*(-b-sqrt(b*b-4*a*c))/(2*a);
    printf("%.2f %.2f",x1,x2);
    return 0;
}