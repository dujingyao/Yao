#include<stdio.h>

int main(){
    int a,b,c;
    scanf("%d %d %d",&a,&b,&c);
    printf("%d %.2f",a+b+c,1.0*(a+b+c)/3);
    return 0;
}