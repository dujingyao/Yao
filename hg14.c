#include<stdio.h>

int main(){
    int i,sum=0;
    while(i<=20){
        sum+=i;
        i++;
    }
    printf("%d",sum);
    return 0;
}