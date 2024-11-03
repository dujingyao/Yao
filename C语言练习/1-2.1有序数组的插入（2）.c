//方法二——类似二分法
#include<stdio.h>
#include<stdlib.h>
#include <stdbool.h>
#define NotFound -1
typedef int Position;
typedef struct LNode * List;
struct LNode{
    int data[5];
    Position Last;
    Position Left;
    Position Right;
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
    if (L->Last >= 4) {
    printf("数组已满，无法插入\n");
    return false;
    }
    int Mid,low=0,high=L->Last;
    while(low<=high){
        Mid=(low+high)/2;
        if(L->data[Mid]==X) return false;
        if(L->data[Mid]>X) low=Mid+1;
        else high=Mid-1;
    }
    int i,j;
    for(i=L->Last;i>=low;i--){
        L->data[i+1]=L->data[i];
    }
    L->data[low]=X;
    ++L->Last;
    return true;
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