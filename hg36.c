#include<stdio.h>

int main(){
    int n;
    scanf("%d",&n);
    int h,min;
    h=n/60,min=n%60;
    printf("%d %d",h,min);
    return 0;
}