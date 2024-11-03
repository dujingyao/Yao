#include<stdio.h>
char upper(char ch);
int main(){
    char ch;
    scanf("%c",&ch);
    printf("%c",upper(ch));
    return 0;
}
char upper(char ch){
    ch-=32;
    return ch;
}