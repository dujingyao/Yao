#include<stdio.h>

int main(){
    int n;
    scanf("%d",&n);
    int a,b,c;
    a=n/100;
    b=n%100/10;
    c=n%10;
    printf("%d",a+b+c);
    return 0;
}