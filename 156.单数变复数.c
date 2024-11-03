#include<stdio.h>
#include<string.h>
int main(){
    char a[25];
    gets(a);
    int i=0,len;
    len=strlen(a);
    if(a[len-1]=='y'){
        a[len-1]='i';
        a[len]='e';
        a[len+1]='s';
        a[len+2]='\0';
        i=1;
    }
    else if(a[len-1]=='s'||a[len-1]=='x'||(a[len-2]=='c'&&a[len-1]=='h')||(a[len-2]=='s'&&a[len-1]=='h')){
        a[len]='e';
        a[len+1]='s';
        a[len+2]='\0';
        i=1;
    }
    if(a[len-1]=='o'){
        a[len]='e';
        a[len+1]='s';
        a[len+2]='\0';
        i=1;
    }
    if(i==0){
        a[len]='s';
        a[len+1]='\0';
    }
    for(int j=0;a[j]!='\0';j++){
        printf("%c",a[j]);
    }

    return 0;
}