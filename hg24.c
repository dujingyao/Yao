#include<stdio.h>
#include<string.h>
int main(){
    char a[11];
    gets(a);
    for(int i=0;i<strlen(a);i++){
        if(a[i]<='z'&&a[i]>='a'){
            a[i]-=32;
        }
    }
    printf("%s",a);
    return 0;
}