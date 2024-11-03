#include<stdio.h>

int main(){
    int i,n=0;
    for(i=1;i<=1000;i++){
        if(i%3==0&&i%7==0&&i%5==0){
            printf("%d ",i);
            n++;
        }
        if(n==4){
            printf("\n");
            n=0;
        }
    }
    return 0;
}