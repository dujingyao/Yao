#include<stdio.h>
//构建串的节点
typedef char * String;
int Index(String S,String T,int pos){
    int i=pos,j=1;
    while(i<=S[0]&&j<=T[0]){
        if(S[i]==T[j]){
            i++;
            j++;
        }
        else{
            i=i-j+2;
            j=1;
        }
    }
    if(j>T[0]) return i-T[0];
    return 0;
}
int main(){
    
    return 0;
}