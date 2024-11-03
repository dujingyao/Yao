#include<stdio.h>

int main(){
    int n;
    scanf("%d",&n);
    if(n<30) printf("%d",n*25);
    if(n>=30) printf("%d",n*20);
    return 0;
}