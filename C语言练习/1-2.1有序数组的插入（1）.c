//方法一
#include<stdio.h>
#include<stdlib.h>
#include <stdbool.h>
typedef int Position;
typedef struct LNode * List;
struct LNode{
    int data[5];
    Position Last;
};
List CreateList(){
    List L=(List)malloc(sizeof(struct LNode));
    if(L==NULL){
        printf("内存分配失败\n");
        exit(1);
    }
    L->Last=3;
    L->data[0]=5;
    L->data[1]=4;
    L->data[2]=3;
    L->data[3]=1;
    return L;
}
bool Insert(List L,int X){
    int i,j;
    for(i=L->Last;i>=0;i--){
        if(L->data[i]<X&&L->data[i-1]>X||i==0&&X>L->data[0]){
            ++L->Last;
            for(j=L->Last;j>i;j--){
                L->data[j]=L->data[j-1];
            }
            L->data[i]=X;
            return true;
        }
        if(X<L->data[L->Last]){
            ++L->Last;
            L->data[L->Last]=X;
            return true;
        }
        if(L->data[i]==X)
            return false;
    }
    return false;
}
int main(){
    List L=CreateList();
    int X;
    scanf("%d",&X);
    if(Insert(L,X)){
        printf("插入成功，插入后序列如下：");
        int i;
        for(i=0;i<=L->Last;i++){
            printf("%d ",L->data[i]);
        }
        printf("\n");
    }else{
        printf("插入失败");
    }
    
    return 0;
}