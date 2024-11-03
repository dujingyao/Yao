#include<stdio.h>

int main(){
    int x;
    scanf("%d",&x);
    int y;
    if(x>0) y=18;
    if(x==0) y=0;
    if(x<0) y=-18;
    printf("%d",y);
    return 0;
}