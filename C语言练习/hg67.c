#include<stdio.h>

int main(){
    int a,b,c,n;
    scanf("%d %d %d",&a,&b,&c);
    if(a<b){
        n=a;
        a=b;
        b=n;
    }
    if(b<c){
        n=b;
        b=c;
        c=n;
    }
    if(a<b){
        n=a;
        a=b;
        b=n;
    }
    printf("%d %d %d",a,b,c);
    return 0;
}