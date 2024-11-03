#include<stdio.h>
#include <stdlib.h>
#define OK 1
#define ERROE 0
//创建一个节点
typedef struct Node{
    int data;
    struct Node *next;
}Node;
typedef Node *Linklist;
//初始化单链表
Linklist initList(){
    Linklist head;
    head=(Linklist)malloc(sizeof(Node));
    head->next=NULL;
    return head;
}
//使用头插法插入节点
void CreatListHead(Linklist head,int data){
    Linklist newNode=(Linklist)malloc(sizeof(Node));
    newNode->next=head->next;//从头部插入
    head->next=newNode;
    newNode->data=data;
}
//使用尾插法插入节点
void CreatListTail(Linklist head,int data){
    Linklist newNode=(Linklist)malloc(sizeof(Node));
    newNode->data=data;
    newNode->next=NULL;
    Linklist Temp;
    Temp=head;
    while(Temp->next!=NULL){
        Temp=Temp->next;
    }
    Temp->next=newNode;
}
int main(){
    Linklist L1,L2,L3;//创建三个头结点
    L1=initList();
    L2=initList();
    L3=initList();
    int data1,data2;
    printf("输入L1链表元素,以-1结束:\n");
    while(scanf("%d",&data1)&&data1!=-1){
        CreatListTail(L1,data1);
    }
    printf("输入L2链表元素,以-1结束:\n");
    while(scanf("%d",&data2)&&data2!=-1){
        CreatListTail(L2,data2);
    }//链表插入数据
    Linklist temp1,temp2,temp3;
    temp1=L1->next;
    temp2=L2->next;
    temp3=L3;//
    while(temp1!=NULL&&temp2!=NULL){
        if(temp1->data<temp2->data){
            temp1=temp1->next;
        }
        else if(temp1->data>temp2->data){
            temp2=temp2->next;
        }
        else{
            Linklist newNode=(Linklist)malloc(sizeof(Node));
            newNode->data=temp1->data;
            newNode->next=NULL;
            temp3->next=newNode;//将新节点连接到L3中
            temp3=newNode;//重置L3
            temp1=temp1->next;
            temp2=temp2->next;
        }
    }
    printf("交集链表元素:\n");
    Linklist temp=L3->next;
    if(temp==NULL) printf("NULL\n");
    while(temp!=NULL){
        printf("%d ",temp->data);
        temp=temp->next;
    }

    return 0;
}