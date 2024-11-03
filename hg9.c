#include<stdio.h>

int main(){
    int a=1,b=1,c;
    printf("1\n1\n");
    for(int i=3;i<=20;i++){
        c=a+b;
        printf("%d\n",c);
        a=b;
        b=c;
    }
    return 0;
}