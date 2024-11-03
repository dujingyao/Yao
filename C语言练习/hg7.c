#include<stdio.h>

int main(){
    int m,n;
    while(scanf("%d %d",&m,&n)!=EOF){
        if(m>n) printf("%d\n",m);
        else printf("%d\n",n);
    }
    return 0;
}