#include<stdio.h>
#include<string.h>
int main(){
    char a[100];
    gets(a);
    printf("%d",strlen(a));
    return 0;
}