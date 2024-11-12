#include<iostream>
#include<algorithm>
#include<cstring>
using namespace std;

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

//计算二叉树的结点数
int NodeCount(BiTree T){
    if(T==NULL) return 0;
    else return NodeCount(T->Lchild)+NodeCount(T->Rchild)+1;
}

//二叉树的深度
int Depth(BiTree T){
    if(T==NULL) return 0;
    else{
        m=Depth(T->Lchild);
        n=Depth(T->Rchild);
        if(m>n) return (m+1);
        else return (n+1);
    }
}

int main(){
    
    return 0;
}