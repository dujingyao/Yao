#include<stdio.h>
#include<string.h>
int main(){
    char a[100];
    int i,sum=0;
    for(i=1;i<=12;i++){
        scanf("%s",a);
        if(i==strlen(a)){
            sum++;
        }
    }
    printf("%d",sum);
    return 0;
}