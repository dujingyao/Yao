#include<stdio.h>

int main(){
    int n,m,k;
    scanf("%d",&n);
    int a[n];
    for(int i=0;i<n;i++){
        scanf("%d",&a[i]);
    }
    scanf("%d",&m);
    for(int i=0;i<m;i++){
        scanf("%d",&k);
        for(int j=0;j<n;j++){
            if(k==a[j]){
                printf("%d\n",j);
                break;
            }
            if(j==n-1) printf("Not found!\n");
        }
    }

    return 0;
}