#include<stdio.h>
#include<string.h>
int main(){
    char a[1000];
    int n,i,sum=0,x=1,j;
    scanf("%s",a);
    int len=strlen(a);
    scanf("%d",&n);
    for(i=0;i<len-n+1;i++){
        x=1;
        if(a[i]=='1'){
            for(j=i+1;j<i+n;j++){
                if(a[j]=='1') x++;
            }
            if(x==n) sum++;
        }
    }
    printf("%d",sum);
    return 0;
}