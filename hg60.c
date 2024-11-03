#include<stdio.h>

int main(){
    int a[10],sum=0;
    for(int i=0;i<10;i++){
        scanf("%d",&a[i]);
        if(i%2==0) sum+=a[i];
    }
    printf("%d",sum);
    return 0;
}