#include<stdio.h>
#include<stdlib.h>
typedef struct Node *PtrToNode;
struct Node{
    int data;
    PtrToNode next;
};
typedef PtrToNode List;
List Insert(List L,int x){
    List pre,tmp;
    pre=L;
    while(pre->next){
        if(x<pre->next->data) break;
        else pre=pre->next;
    }
    tmp=(PtrToNode)malloc(sizeof(struct Node));
    tmp->data=x;
    tmp->next=pre->next;
    pre->next=tmp;
    return L;
}

int main(){
    
    return 0;
}