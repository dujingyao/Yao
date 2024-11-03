#include<stdio.h>
typedef struct Node *PtrToNode;
struct Node{
    int data;
    PtrToNode next;
};
typedef PtrToNode List;
List Merge(List L1,List L2){
    List L3,rear;
    L3=(PtrToNode)malloc(sizeof(struct Node));
    L3->next=NULL;
    rear=L3;
    while(L1->next!=NULL&&L2->next!=NULL){
        if(L1->next->data<L2->next->data){
            rear->next=L1->next;
            L1->next=L1->next->next;
            rear->next->next=NULL;
            rear=rear->next;
        }
        else{
            rear->next=L2->next;
            L2->next=L2->next->next;
            rear->next->next=NULL;
            rear=rear->next;
        }
    }
    if(L1->next){
        rear->next=L1->next;
        L1->next=NULL;
    }
    if(L2->next){
        rear->next=L2->next;
        L2->next=NULL;
    }
    return L3;
}
int main(){
    
    
    return 0;
}