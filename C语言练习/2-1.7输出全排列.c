#include<stdio.h>
#define MAXN 9
void RightShift(int L[],int Left,int i){
    int j,Tmp=L[i];
    for(j=i;j>Left;j--) L[j]=L[j-1];
    L[j]=Tmp;
}
void LeftShift(int L[],int Left,int i){
    int j,Tmp=L[Left];
    for(j=Left;j<i;j++) L[j]=L[j+1];
    L[j]=Tmp;
}
void Permutation(int L[],int Left,int Right){
    int i;
    if(Left==Right){
        for(i=0;i<=Right;i++) printf("%d",L[i]);
        printf("\n");
    }else{
        for(i=Left;i<=Right;i++){
            RightShift(L,Left,i);
            Permutation(L,Left+1,Right);
            LeftShift(L,Left,i);
        }
    }
}
int main(){
    int N,L[MAXN],i;
    scanf("%d",&N);
    for(i=0;i<N;i++) L[i]=i+1;
    Permutation(L,0,N-1);
    return 0;
}