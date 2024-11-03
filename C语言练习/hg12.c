#include<stdio.h>

int main(){
    int n;
    while(scanf("%d",&n)!=EOF){
        if(n%3==0) printf("Yes\n");
        else printf("No\n");
    }
    return 0;
}