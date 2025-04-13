#include<stdio.h>
#include<stdlib.h>
//构建串的节点
typedef char * String;
//T为子串
void get_next(String T,int *next){
    int i,k;
    i=1;
    k=0;
    next[1]=0;
    while(i<T[0]){
        if(k==0||T[i]==T[k]){
            i++;
            k++;
            next[i]=k;
        }
        else{
            k=next[k];
        }
    }
}
int Index(String S,String T,int pos){
    int i=pos,j=1;
    int next[225];
    get_next(T,next);
    while(i<=S[0]&&j<=T[0]){
        if(j==0||S[i]==T[j]){
            i++;
            j++;
        }
        else{
            j=next[j];//i不变,j回溯到需要改变的地方
        }
    }
    if(j>T[0]) return i-T[0];
    else return 0;
}
int main(){
    String S=(String)malloc(6*sizeof(char));//0号位为字符串的长度
    S[0]=5;S[1]='a';S[2]='a';S[3]='c';S[4]='b';S[5] ='a';
    String T=(String)malloc(3*sizeof(char));
    T[0]=2;T[1]='b';T[2]='a';

    int i=Index(S,T,1);
    printf("%d\n",i);

    free(S);
    free(T);

    return 0;
}