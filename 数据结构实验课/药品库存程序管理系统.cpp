#include<stdio.h>
#include<string.h>
#include<math.h>
#define OK 1
#define ERROR 0
#define MAXSIZE 1000
#define OVERFLOW -1
//药品库存数据结构体
//结点
typedef struct MEnode{
    char id[10];                //药品编号
    char name[10];             //药品名称
    int number;                //药品数量
}MEnode;
//总体的顺序表
typedef struct{
    MEnode * newME;            //存储药品的基地址
    int length;                //最大长度
}MElist;
//初始化
int Init(MElist &l){
    l.newME=new MEnode[MAXSIZE];
    if(!l.newME) return OVERFLOW;
    l.length=0;
    return OK;
}
//建立(插入初始数据操作)
void Create(MElist &l,MEnode e){
    l.newME[l.length]=e;
    l.length++;
}
//插入操作
int Insert(MElist &L,int i,MEnode e){
    if(i>L.length){
        return ERROR;
    }
    int j;
    for(j=L.length-1;j>=i-1;j--){
        L.newME[j+1]=L.newME[j];
    }
    L.length++;
    L.newME[i-1]=e;
    return OK;
}
//删除操作
int Delete(MElist &L,int i){
    if(i>L.length){
        return ERROR;
    }
    int j;
    for(j=i-1;j<L.length-1;j++){
        L.newME[j]=L.newME[j+1];
    }
    L.length--;
    return OK;
}
//比较函数
int compare(MEnode a,MEnode b){
    int len_id1=strlen(a.id);
    int len_id2=strlen(b.id);
    if(len_id1!=len_id2) return 0;
    int len_name1=strlen(a.name);
    int len_name2=strlen(b.name);
    if(len_name1!=len_name2) return 0;
    int i,j;
    for(i=0;i<len_id1;i++){
        if(a.id[i]!=b.id[i]) return 0;
    }
    for(j=0;j<len_name1;j++){
        if(a.name[j]!=b.name[j]) return 0;
    }
    return 1;
}
//查找操作
int Locate(MElist L,MEnode e){
    int i;
    for(i=0;i<L.length;i++){
        if(compare(L.newME[i],e)) return i+1;
    }
    return ERROR;
}
//取值操作
int Get(MElist L,int i,MEnode &e){
    if(i<1||i>L.length){
        return ERROR;
    }
    e=L.newME[i-1];
    return OK;
}
//遍历显示全部内容
void f(MElist L){
    int i;
    for(i=0;i<L.length;i++){
        printf("药品的编号为:%s,药品的名称为:%s,药品的库存为:%d\n",L.newME[i].id,L.newME[i].name,L.newME[i].number);
    }
}
//库存变化
void inventory(MElist &L,int i,int n){
    if(i>L.length){
        printf("序号输入有误\n");
        return;
    }
    if(n<0&&abs(n)>L.newME[i-1].number){
        printf("库存不足!");
        return;
    }
    L.newME[i-1].number=L.newME[i-1].number+n;
}

int main(){
    int N;
    MElist L;
    MEnode newME;
    if(!(Init(L))){
        printf("分配失败\n");
        return 0;
    }
    printf("请输入药品种类数量:");
    scanf("%d",&N);          //有N个药品
    int i;
    L.length=0;
    //将商品的特征传到L中
    for(i=0;i<N;i++){
        printf("第%d个药品的编号为:",i+1);
        scanf("%s",newME.id);
        printf("第%d个药品的名称:",i+1);
        scanf("%s",newME.name);
        printf("第%d个药品的库存量:",i+1);
        scanf("%d",&newME.number);
        Create(L,newME);
    }
    printf("请选择操作:插入请输入1,删除请输入2,查找请输入3,取值请输入4.\n");
    printf("请输入操作:");
    int caozuo;
    scanf("%d",&caozuo);
    if(caozuo==1){            //实行插入操作
        int n;
        printf("输入要插入的位置:");
        scanf("%d",&n);
        MEnode e;
        printf("药品的编号为:");
        scanf("%s",e.id);
        printf("药品的名称:");
        scanf("%s",e.name);
        printf("药品的库存量");
        scanf("%d",&e.number);
        if(Insert(L,n,e)) printf("插入成功!\n");
        else printf("插入失败!\n");
    }
    if(caozuo==2){          //实行删除操作
        int n;
        printf("请输入要删除的位置:");
        scanf("%d",&n);
        if(Delete(L,n)) printf("删除成功!\n");
        else printf("删除失败!\n");
    }
    if(caozuo==3){          //实行查找操作
        MEnode e;
        printf("药品的编号为:");
        scanf("%s",e.id);
        printf("药品的名称:");
        scanf("%s",e.name);
        printf("药品的库存量");
        scanf("%d",&e.number);
        if(Locate(L,e)) printf("查找成功,该药品的位置为:%d\n",Locate(L,e));
        else printf("查找失败\n");
    }
    if(caozuo==4){             //实施取值操作
        int n;
        printf("输入取值的位置:");
        scanf("%d",&n);
        MEnode e;
        if(Get(L,n,e)) printf("取值成功:该位置的编号为:%s,该位置的名称为%s,该位置的库存为:%d\n",e.id,e.name,e.number);
        else printf("取值失败.\n");
    }
    if(caozuo>4&&caozuo<-1) printf("请输入正确的选项.\n");
    f(L);                   //遍历数组
    printf("库存是否变化:有变化输入1,无变化输入2\n");
    int kucun;
    scanf("%d",&kucun);
    if(kucun==1){
        int i,n;
        printf("请输入变化的位置:");
        scanf("%d",&i);
        printf("请输入库存变化情况:");
        scanf("%d",&n);
        inventory(L,i,n);
    }
    f(L);                   //遍历数组

    return 0;
}