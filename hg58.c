#include<stdio.h>

int main(){
    int n;
    scanf("%d",&n);
    while(1){
        if(n<10){
            printf("%d",n);
            break;
        }
        printf("%d ",n%10);
        n/=10;
    }
    return 0;
}