#include<stdio.h>
//队列
#define MAXSIZE 100
#define OK 1
#define ERROR -1
typedef int QElemType;
//循环队列的顺序存储结构
typedef struct{
    QElemType data[MAXSIZE];
    int front;
    int rear;
}SqQueue;
//循环队列的初始化
//初始化一个空队列
int InitiQueue(SqQueue *Q){
    Q->front=0;
    Q->rear=0;
    return OK;
}
//循环队列求队列长度
int QueueLength(SqQueue Q){
    return (Q.rear-Q.front+MAXSIZE)%MAXSIZE;
}
//顺序队列入队操作
int EnQueue(SqQueue *Q,QElemType e){
    if((Q->rear+1)%MAXSIZE==Q->front)/*队列满*/
        return ERROR;
    Q->data[Q->rear]=e;
    Q->rear=(Q->rear+1)%MAXSIZE;//若到最后则转到数组头部
    return OK;
}
//循环队列出队操作
int DeQueue(SqQueue *Q,QElemType *e){
    if(Q->front==Q->rear)
        return ERROR;
    *e=Q->data[Q->front-1];
    Q->front=(Q->front+1)%MAXSIZE;
    return OK;
}
//链队列


int main(){
    
    return 0;
}