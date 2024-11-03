#include<stdio.h>

int main(){
    int i,n=0;
    for(i=1;i<=100;i++){
        if(i%8==0){
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