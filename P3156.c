#include<stdio.h>

int main(){
    int n,m;
    scanf("%d %d",&n,&m);
    int a[n],b[m];
    for(int j=0;j<n;j++){
        scanf("%d",&a[j]);
    }
    for(int i=0;i<m;i++){
        scanf("%d",&b[i]);
    }
    int i=0;
    while(m--){
        printf("%d\n",a[b[i]-1]);
        i++;
    }
    return 0;
}