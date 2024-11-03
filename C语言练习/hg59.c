#include<stdio.h>

int main(){
    int n,q=1;
    scanf("%d",&n);
    for(int i=1;i<=n;i++){
        q*=i;
    }
    printf("%d",q);
    return 0;
}