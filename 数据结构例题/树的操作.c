#include<stdio.h>
#define MAXSIZE 100
#define OK 1
//双亲表示法
typedef struct PTNnode{            //结点结构
    int data;
    int parent;                    //双亲位置
}PTNnode;
typedef struct{                    //树结构
    PTNnode node[MAXSIZE];         //结点数组
    int r,n;                       //根的位置和结点数
}PTerr;
//孩子表示法
typedef struct CTNode{             //孩子结点
    int child;
    struct CTNode *next;
}*ChildPtr;
typedef struct{                     //表头结点
    int data;
    ChildPtr fristchild;
}CTBox;
typedef struct{                     //树结构
    CTBox nodes[MAXSIZE];           //结点数组
    int r,n;                        //根的位置和结点位置
}CTree;
//双亲孩子表示法


//孩子兄弟表示法
typedef struct CSNode{
    int data;
    struct CSNode *fristchild,*rightsib;
}CSNode,*CSTree;

//二叉树的结构体
//二叉链表
typedef struct BiNode{
    char data;
    BiNode * Lchild;
    BiNode * Rchild;
}BiNode,*BiTree;
//二叉树的建立
void CreateBiTree(BiTree * T){
    char ch;
    scanf("%c",&ch);
    if(ch=='#') *T=NULL;//说明该路径已经填充完
    else{
        *T=(BiTree)malloc(sizeof(BiNode));
        if(!*T) exit(-1);
        (*T)->data=ch;
        CreateBiTree(&(*T)->Lchild);//构建左子树
        CreateBiTree(&(*T)->Rchild);//构建右子树
    }
}

//二叉树的遍历
//前序遍历
void PreOrderTraverse(BiTree T){
    if(T==NULL) return;
    printf("%c",T->data);
    PreOrderTraverse(T->Lchild);
    PreOrderTraverse(T->Rchild);
}
//中序遍历
void InOrderTraverse(BiTree T){
    if(T==NULL) return;
    InOrderTraverse(T->Lchild);
    printf("%c",T->data);
    InOrderTraverse(T->Rchild);
}
//后序遍历
void PostOrderTraverse(BiTree T){
    if(T==NULL) return;
    PostOrderTraverse(T->Lchild);
    PostOrderTraverse(T->Rchild);
    printf("%c",T->data);
}

//线索二叉树
//结构
typedef enum {Link,Thread} PointerTag; 
//Link=0表示指向左右孩子指针
//Thread=1表示指向前驱或后继的线索
typedef struct BiThrNode{
    char data;
    struct BiThrNode *lchild,*rchild;
    PointerTag ltag;
    PointerTag rtag;
}BiThrNode,*BiThrTree;
//中序遍历线索化
BiThrTree pre;
void InThreading(BiThrTree p){
    if(p){
        InThreading(p->lchild);    //递归左子树线索化
        if(!p->lchild){
            p->ltag=Thread;
            p->lchild=pre;
        }
        if(!pre->rchild){
            pre->rtag=Thread;
            pre->rchild=p;
        }
        pre=p;
        InThreading(p->rchild);     //递归右子树线索化
    }
}
//T是指向头结点的指针,头结点的左孩子指向根节点,右孩子指向遍历的最后一个节点
int InOrderTraverse_Thr(BiThrTree T){
    BiThrTree P;
    P=T->lchild;
    while(P!=T){
        while(P->ltag==Link){//等于Link意味着有孩子结点
            P=P->lchild;
        }
        printf("%c",P->data);//打印第一个结点
        while(P->rtag==Thread&&P->rchild!=NULL){
            P=P->rchild;
            printf("%c",P->data);
        }
        P=P->rchild;
    }
    return OK;
}

int main(){
    
    return 0;
}