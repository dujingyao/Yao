#include<stdio.h>

int main(){
    int a,b,c,i;
    for(i=100;i<=200;i++){
        a=i/100;
        b=i%100/10;
        c=i%10;
        if(a+b+c==6) printf("%d\n",i);
    }
    return 0;
}