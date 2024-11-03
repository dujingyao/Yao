#include<stdio.h>
#include <stdlib.h>
#define NotFound -1
typedef int Position;
typedef struct LNode *List;
struct LNode{
    int data[5];
    Position Last;
};
Position BS(List L,int X,Position Left,Position Right){
    if(Left>Right) return NotFound;//判断是否为空
    Position Mid;
    Mid=(Left+Right)/2;
    if(L->data[Mid]>X) return BS(L,X,Left,Mid-1);
    else if(L->data[Mid]<X) return BS(L,X,Mid+1,Right);
    return Mid;
}
Position BinarySearch(List L,int X){
    return BS(L,X,0,L->Last);
}
List CreateList(){
    List L=(List)malloc(sizeof(struct LNode));
    if(L==NULL){
        printf("内存分配失败,无法创建链表");
        return NULL;
    }
    L->Last=5;
    L->data[0]=1;
    L->data[1]=2;
    L->data[2]=3;
    L->data[3]=4;
    L->data[4]=5;
    return L;
}
int main(){
    List L=CreateList();
    if(L==NULL){
        return 1;
    }
    int X;
    scanf("%d",&X);
    Position index=BinarySearch(L,X);
    if(index==NotFound){
        printf("元素%d不在链中\n",X);
    }else{
        printf("元素%d在链表中的索引为%d",X,index);
    }
    
    return 0;
}