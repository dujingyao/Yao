#include<stdio.h>
#define N 30
int f=0;
int Terms[N];
//目前已经分解到第nTerm项
//对剩余值Reamainder继续分解
//要求之后的每个分解项都大于Start
void Search(int Remainder,int Start,int nTerm,int n);
int main(){
    int n;
    scanf("%d",&n);
    Search(n,1,0,n);
    return 0;
}
void Search(int Remainder,int Start,int nTerm,int n){
    if(Remainder==0){
        f++;
        printf("%d=",n);
        int i;
        for(i=0;i<nTerm-1;i++){
            printf("%d+",Terms[i]);
        }
        printf("%d;",Terms[i]);
        if(f%4==0) printf("\n");
    }
    for(int j=Start;j<=Remainder;j++){
        Terms[nTerm]=j;
        Search(Remainder-j,j,nTerm+1,n);
    }
}