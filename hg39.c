#include<stdio.h>

int main(){
    double sum=0.0,flag=1.0;
    for(int i=1;i<=100;i++){
        sum+=(flag*1.0/i);
        flag=-flag;
    }
    printf("%.4f",sum);
    return 0;
}