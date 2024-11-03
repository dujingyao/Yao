#include<stdio.h>
int seek(char ch[],char a,char b){
    int x=1,y=1;
    int f1,f2,f;
    while(ch[x]!=a){
        x++;
    }
    while(ch[y]!=b){
        y++;
    }
    if(x>=y) f1=26-x+y;
    else f2=y-x;
    if(f1>f2) f=f2;
    else f=f1;
    return f;
}
int main(){
    //输入26个字母
    char ch[27],a[27];
    int sum=0;
    for(int i=1;i<=26;i++){
        scanf("%c",&ch[i]);
    }
    //找A
    for(int i=1;i<=26;i++){
        if(ch[i]=='A'){
            sum+=i;
            break;
        }
    }
    //生成字母表
    a[1]='A';
    for(int i=2;i<=26;i++){
        a[i]=a[i-1]+1;
    }    
    int n;
    for(int i=1;i<=25;i++){
        n=seek(ch,a[i],a[i+1]);
        sum+=n;
    }
    printf("%d",sum);
    return 0;
}