#include<stdio.h>

int main(){
    int t,n,a[10010],len,lenmax,i,j,k;
    scanf("%d",&t);
    while(t--){
        scanf("%d",&n);
        len=1;
        lenmax=1;
        for(i=0;i<n;i++){
            scanf("%d",&a[i]);
        }
        for(i=0;i<n;i++){
            for(j=i+1;j<n;j++){
                if(a[j]<a[i]){
                    k=a[i];
                    a[i]=a[j];
                    a[j]=k;
                }
            }
        }
        for(i=1;i<n;i++){
            if(a[i-1]==a[i]) len++;
            else len=1;
            if(len>lenmax) lenmax=len;
        }
        printf("%d\n",lenmax);
    }
    return 0;
}