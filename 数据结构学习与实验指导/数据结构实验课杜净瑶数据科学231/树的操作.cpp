#include<iostream>
#include<algorithm>
#include<cstring>
using namespace std;

typedef struct BiNode{
    char data;
    BiNode * Lchild;
    BiNode * Rchild;
    bool ltag, rtag; // 线索标志位，true 表示线索，false 表示孩子指针
}BiNode,*BiTree;
//二叉树的建立
void CreateBiTree(BiTree * T){
    char ch;
    scanf(" %c",&ch);
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
        int m=Depth(T->Lchild);
        int n=Depth(T->Rchild);
        if(m>n) return (m+1);
        else return (n+1);
    }
}
// 线索化函数
void ThreadInOrder(BiTree p, BiTree &pre) {
    if(p!=NULL) {
        ThreadInOrder(p->Lchild, pre);
        if(p->Lchild==NULL) {
            p->Lchild=pre;
            p->ltag=true;
        }
        if(pre!=NULL&&pre->Rchild==NULL) {
            pre->Rchild=p;
            pre->rtag=true;
        }
        pre=p;
        ThreadInOrder(p->Rchild, pre);
    }
}

// 创建线索二叉树
void CreateThreadedBinaryTree(BiTree T) {
    BiTree pre=NULL;
    if(T!=NULL) {
        ThreadInOrder(T, pre);
        pre->Rchild=NULL; // 处理最后一个节点的右线索
        pre->rtag=true;
    }
}

// 中序遍历线索二叉树
void InOrderTraverseThreaded(BiTree T) {
    BiTree p = T;
    while(p != NULL) {
        // 找到最左边的节点
        while(!p->ltag) {
            p = p->Lchild;
        }
        printf("%c", p->data); // 访问节点

        // 利用右线索移动到下一个节点
        while(p->rtag && p->Rchild != NULL) {
            p = p->Rchild;
            printf("%c", p->data);
        }
        p = p->Rchild;
    }
}


int main(){
    BiTree T = NULL;
    cout << "请输入二叉树的先序序列（#表示空节点）：";
    CreateBiTree(&T);

    cout << "前序遍历: ";
    PreOrderTraverse(T);
    cout << endl;

    cout << "中序遍历: ";
    InOrderTraverse(T);
    cout << endl;

    cout << "后序遍历: ";
    PostOrderTraverse(T);
    cout << endl;

    cout << "结点数: " << NodeCount(T) << endl;
    cout << "深度: " << Depth(T) << endl;

    // 创建线索二叉树
    CreateThreadedBinaryTree(T);

    cout << "线索二叉树的中序遍历: ";
    InOrderTraverseThreaded(T);
    cout << endl;
    return 0;
}