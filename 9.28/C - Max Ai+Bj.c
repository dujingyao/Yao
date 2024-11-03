#include<stdio.h>
int max(int a[],int N){
    int max=a[1];
    for(int i=1;i<=N;i++){
        if(a[i]>max) max=a[i];
    }
    return max;
}
int main(){
    int N;
    scanf("%d",&N);
    int a[N+1],b[N+1];
    for(int i=1;i<=N;i++){
        scanf("%d",&a[i]);
    }
    for(int i=1;i<=N;i++){
        scanf("%d",&b[i]);
    }
    int max1=max(a,N);
    int max2=max(b,N);
    printf("%d",max1+max2);
    return 0;
}