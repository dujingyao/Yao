#include<stdio.h>

int main(){
    int a[10],sum=0;
    for(int i;i<10;i++){
        scanf("%d",&a[i]);
        sum+=a[i];
    }
    printf("%d",sum);
    return 0;
}