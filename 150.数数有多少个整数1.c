#include<stdio.h>
#include<ctype.h>
int main(){
    
    char a[1000];
    int i,sum=0,flag=0;
    gets(a);
    for(i=0;a[i]!='\0';i++){
        if(isdigit(a[i])&&flag==0){
            if(a[i]=='0'&&isdigit(a[i+1])){
                sum++;
                flag=0;
                continue;
            }
            sum++;
            flag=1;
        }
        if(!isdigit(a[i])){
            flag=0;
        }
    }
    printf("%d\n",sum);
    return 0;
}