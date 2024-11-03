#include<stdio.h>

int main(){
    int N;
    scanf("%d",&N);
    int a[N];
    int i;
    for(i=0;i<N;i++){
        scanf("%d",&a[i]);
    }  
    int b[10000],j,sum=0;
    for(i=0;i<N;i++){
        for(j=0;j<N;j++){
            if(i==j) continue;
            sum+=(a[i]*10+a[j]);
        }
    }
    printf("%d",sum);
    return 0;
}