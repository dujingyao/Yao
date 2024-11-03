#include<stdio.h>
#define ERROR -1
#define OK 1
//栈
//栈的顺序结构
//栈的结构定义
typedef int SELemType;
typedef struct{
    SELemType data[100];
    int top;
}SqStack;
//进栈操作
int Push(SqStack *S,SELemType e){
    if(S->top==100-1){//栈满
        return ERROR;
    }
S->top++;
S->data[S->top]=e;
}
//出栈操作
int Pop(SqStack *S,SELemType *e){
    if(S->top==-1)
        return ERROR;
    *e=S->data[S->top];
    S->top--;
    return OK;
}
//两栈共享空间结构
typedef struct{
    SELemType data1[100];
    int top1;
    int top2;
}SqDoubleStack;

int Push1(SqDoubleStack *S,SELemType e,int stackNumber){
    if(S->top1+1==S->top2)
        return ERROR;
    if(stackNumber==1){
        S->data1[++S->top1]=e;
    }
    else if(stackNumber==2){
        S->data1[--S->top2]=e;
    }
    return OK;
}
//两栈共享的pop
int Pop1(SqDoubleStack *S,SELemType *e,int stackNumber){
    if(stackNumber==1){
        if(S->top1==-1)
            return ERROR;
        *e=S->data1[S->top1--];
    }
    else if(stackNumber==2){
        if(S->top2==-1)
            return ERROR;
        *e=S->data1[S->top2++];
    }
    return OK;
}
//栈的链式结构
typedef struct StackNode{
    SELemType data;
    struct StackNode *next;
}StackNode,*LinkStackper;
typedef struct{
    LinkStackper top;
    int count;
}LinkStack;
//链栈的push
int Push(LinkStack *S,SELemType e){
    LinkStackper s=(LinkStackper)malloc(sizeof(StackNode));//分配一个节点储存空间
    s->data=e;
    s->next=S->top;
    S->top=s;
    S->count++;
    return OK;
}
//链栈的pop
int Pop2(LinkStack *S,SELemType *e){
    LinkStackper p;
    if(StackEmpty(*S))
        return ERROR;
    *e=S->top->data;
    p=S->top;
    S->top=S->top->next;
    free(p);
    S->count--;
    return OK;
}

int main(){
    
    return 0;
}