#include<stdio.h>

int main(){
    int n,a,b,x,y,i;
    scanf("%d",&n);
    while(n--){
        int c[10000]={0};
        scanf("%d %d",&a,&b);
        while(b--){
            scanf("%d %d",&x,&y);
            for(i=x;i<=y;i++){
                c[i]=1;
            }
        }
        int sum=0;
        for(i=0;i<=a;i++){
            if(c[i]==0) sum++;
        }
        printf("%d\n",sum);
    }
    return 0;
}