#include<stdio.h>
#include<string.h>
int main(){
    char a[2000],b[1000];
    gets(a);
    gets(b);
    strcat(a,b);
    puts(a);
    return 0;
}