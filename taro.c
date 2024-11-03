#include<stdio.h>

int main(){
    int n,m;
    char c;
    int i,j;
    scanf("%d %d",&n,&m);
    int a[n];
    for(i=0;i<n;i++){
        a[i]=0;
    }
    while(m--){
        scanf("%d %c",&j,&c);
        if(c=='F'){
            printf("No\n");
            continue;
        }
        a[j-1]++;
        if(a[j-1]==1) printf("Yes\n");
        else printf("No\n");
    }
    
    return 0;
}