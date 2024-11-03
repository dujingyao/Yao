#include<stdio.h>
#include<string.h>
int len(char *sp);
int main(){
    char a[1000];
    gets(a);
    printf("%d",len(a));
    return 0;
}
int len(char *sp){
    int i,t=0;
    int len=strlen(sp);
    for(i=0;i<len;i++){
        if(sp[i]!=' ') t++;
    }
    return t;
}