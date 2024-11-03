#include<iostream>
#define MAXVEX 9
#define INFINITY 65535
using namespace std;
//邻接矩阵的存储结构
typedef struct{
    char vex[100];
    int arc[100][100];
    int numNode,numEdges;
}MGraph;
//无向网图的创建代码
void CreatGraph(MGraph *G){
    int i,j,k,w;
    //输入顶点数和边数
    cin>>G->numNode>>G->numEdges;
    //输入顶点信息
    for(i=0;i<100;i++){
        cin>>G->vex[i];
    }
    //初始化邻接矩阵
    for(i=0;i<100;i++){
        for(j=0;j<100;j++){
            G->arc[i][j]=100;
        }
    }
    //建立邻接矩阵
    for(k=0;k<G->numEdges;k++){
        cin>>i>>j>>w;//w是权
        G->arc[i][j]=w;
        G->arc[i][j]=G->arc[j][i];
    }
}

//邻接表
//结点定义
//边表结点
typedef struct EdgeNode{
    int adjvex;//邻接点域
    int info;//权值
    EdgeNode *next;
}EdgeNode;
//顶点表结点
typedef struct VertexNode{
    char data;
    EdgeNode *firstedge;
}VertexNode,AdjList[100];
typedef struct{
    AdjList adjlist;
    int numNode,numEdges;
}GraphAdjList;
//无向图的邻接表的创建
void GreatALGraph(GraphAdjList *G){
    int i,j,k;
    EdgeNode *e;
    cin>>G->numNode>>G->numEdges;//输入结点数和边数
    //输入顶点信息
    for(i=0;i<G->numNode;i++){
        cin>>G->adjlist[i];
        G->adjlist[i].firstedge=NULL;
    }
    //头插法
    for(k=0;k<G->numEdges;k++){
        cin>>i>>j;//输入边(vi,vj)上的顶点序号
        e=new EdgeNode;
        e->adjvex=i;
        e->next=G->adjlist[i].firstedge;
        G->adjlist[i].firstedge=e;
        e=new EdgeNode;
        e->adjvex=j;
        e->next=G->adjlist[j].firstedge;
        G->adjlist[j].firstedge=e;
    }
}
//prim算法生成最小树
void MiniSpanTree_Prim(MGraph G){
    int min,i,j,k;
    int adjvex[MAXVEX];
    int lowcost[MAXVEX];
    lowcost[0]=0;
    adjvex[0]=0;
    for(int i=1;i<G.numNode;i++){
        min=INFINITY;
        j=1;
        k=0;
        while(j<G.numNode){
            if(lowcost[j]!=0&&lowcost[j]<min){
                min=lowcost[j];
                k=j;
            }
            j++;
        }
        printf("(%d %d)\n",adjvex[k],k);
        lowcost[k]=0;
        
    }
}
int main(){
    
    return 0;
}