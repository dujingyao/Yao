#include<stdio.h>

int main(){
    int n;
    scanf("%d",&n);
    int a,b,c;
    a=n/100;
    b=n%100/10;
    c=n%10;
    int sum=a+b+c;
    printf("%d",sum);
    return 0;
}