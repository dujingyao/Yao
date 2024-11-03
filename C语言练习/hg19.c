#include<stdio.h>

int main(){
    int a,b;
    scanf("%d %d",&a,&b);
    while(1){
        if(a>b) a=a-b;
        else if(b>a) b=b-a;
        else break;
    }
    printf("%d",a);
    return 0;
}