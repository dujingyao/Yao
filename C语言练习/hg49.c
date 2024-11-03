#include<stdio.h>
int max(int a,int b);
int main(){
    int a,b;
    scanf("%d %d",&a,&b);
    int c=max(a,b);
    printf("%d",a*b/c);
    return 0;
}
int max(int a,int b){
    while(1){
        if(a>b) a=a-b;
        else if(b>a) b=b-a;
        else break;
    }
    return b;
}