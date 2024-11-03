#include<stdio.h>

int main(){
    int n,i,j,k,flag;
    scanf("%d",&n);
    int a[n],b[n],c[2*n];
    for(i=0;i<n;i++){
        scanf("%d",&a[i]);
    }    
    for(j=0;j<n;j++){
        scanf("%d",&b[j]);
    }
    j=0;
    k=0;
    for(i=0;i<2*n;i++){
        if(a[j]<b[k]){
            c[i]=a[j];
            j++;
        }else{
            c[i]=b[k];
            k++;
        }
        if(j==n){
            flag=0;
            break;
        }
        if(k==n){
            flag=1;
            break;
        }
    }
    i++;
    if(flag==0){
        for(;i<2*n;i++){
            c[i]=b[k];
            k++;
        }
    }
    if(flag==1){
        for(;i<2*n;i++){
            c[i]=a[j];
            j++;
        }
    }
    int mid=(2*n+1)/2;
    printf("%d",c[mid-1]);
    return 0;
}