#include<stdio.h>
typedef struct LNode{
    int data[10000];
    int Last;     //保存线性表中最后一个元素的位置
}LNode,*list;
//平方复杂度
list deldt(list L,int min,int max){
    int i,j;
    for(i=0;i<=L->Last;i++){
        if(L->data[i]>min&&L->data[i]<max){
            //整体左移
            for(j=i+1;j<=L->Last;j++){
                L->data[i-1]=L->data[i];
                L->Last--;
            }
            i--;
        }
    }
    return L;
}
//线性复杂度
list deldtd(list L,int min,int max){
    int i,j,p;
    for(i=0;i<=L->Last;i++){
        if(L->data[i]>=min){
            p=i;
            break;
        }
    }
    for(;i<=L->Last;i++){
        if(!(L->data[i]>min&&L->data[i]<max)){
            L->data[p++]=L->data[i];
        }
    }
    L->Last=p-1;
    return L;
}
int main(){
    
    return 0;
}