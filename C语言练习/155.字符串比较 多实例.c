#include<stdio.h>
#include<string.h>
int compare(char a[],char b[]);
int main(){
    char a[10000]={'\0'};
    char b[10000]={'\0'};
    while(scanf("%s %s",a,b)!=EOF){
        if(compare(a,b)==1) printf("YES");
        else printf("NO");
        printf("\n");
    }
    
    return 0;
}
int compare(char a[],char b[]){
    int i,j;
    int len1=strlen(a),len2=strlen(b);
    for(i=0;i<len1;i++){
        if(a[i]==b[i]+32||a[i]==b[i]-32){
            if(a[i]<b[i]) return 1;//说明a[i]是大写
            if(a[i]>b[i]) return 0;
        }
        else{
            if(a[i]>='A'&&a[i]<='Z') a[i]=a[i]+32;
            if(b[i]>='A'&&b[i]<='Z') b[i]=b[i]+32;
            if(a[i]<b[i]) return 1;
            if(a[i]>b[i]) return 0;
        }
    }
    if(len1==len2) return 0;
    else return 1;
}