#include<stdio.h>

int main(){
    int n,m;
    scanf("%d",&n);
    int a[n];
    scanf("%d",&m);
    int i;
    for(i=0;i<n;i++){
        scanf("%d",&a[i]);
    }
    int t;
    for(int j=0;j<=m+1;j++){
        t=a[n-1];
        for(i=n-1;i>0;i--){
            a[i]=a[i-1];
        }
        a[0]=t;
    }
    for(i=0;i<n;i++){
        printf("%d ",a[i]);
    }
    return 0;
}