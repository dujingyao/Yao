#include<stdio.h>
int func(int x){
    int sum=0;
    while((double)x/10>0){
         sum*=10;
         sum=sum+x%10;
         x/=10;
    }
    return sum;
}
int main(){
    int x,y;
    scanf("%d %d",&x,&y);
    printf("%d",func(x)+func(y));
    return 0;
}