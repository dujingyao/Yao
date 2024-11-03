#include<stdio.h>

int main(){
    int i;
    for(i=300;i<=400;i++){
        if(i%5==0){
            if(i%7!=0){
                printf("%d\n",i);
            }
        }
        if(i%7==0){
            if(i%5!=0){
                printf("%d\n",i);
            }
        }
    }
    return 0;
}