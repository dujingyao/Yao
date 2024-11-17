#include<iostream>
#include<string>
#include<algorithm>
#define OK 1
#define ERROR -1
using namespace std;
//创建节点
typedef struct MEnode{
    string id;
    string name;
    int number;
    MEnode * next;
}MEnode,*LinkMEnode;
//创建栈
//初始化
int InitStack(LinkMEnode &S){
    S=NULL;
    return OK;
}
//入栈
int Push(LinkMEnode &S,MEnode &e){
    MEnode *p;
    p=new MEnode();
    p->id=e.id;
    p->name=e.name;
    p->number=e.number;
    p->next=S;//将该节点放到之前的栈顶
    S=p;//将栈顶更新为新节点
    return OK;
}
//出栈
int pop(LinkMEnode &S,MEnode &e){
    if(S==NULL){
        return ERROR;
    }
    MEnode *p;
    p=new MEnode();
    e.id=S->id;
    e.name=S->name;
    e.number=S->number;
    p=S;
    S=S->next;
    delete p;
    return OK;
}
int compare(MEnode a,MEnode b){
    int len_id1=a.id.size();
    int len_id2=b.id.size();
    if(len_id1!=len_id2) return 0;
    int len_name1=a.name.size();
    int len_name2=b.name.size();
    if(len_name1!=len_name2) return 0;
    int i,j;
    for(i=0;i<len_id1;i++){
        if(a.id[i]!=b.id[i]) return 0;
    }
    for(j=0;j<len_name1;j++){
        if(a.name[j]!=b.name[j]) return 0;
    }
    return OK;
}
//查找
int check(LinkMEnode S,string ch,MEnode &e){
    MEnode *p=S;
    while(p!=NULL){
        if(p->id==ch){
            e.id=p->id;
            e.name=p->name;
            e.number=p->number;
            return OK;
        }
        p=p->next;
    }
    return ERROR;
}
//取值(取栈顶元素)
int Get(LinkMEnode S,MEnode &e){
    if(S==NULL) return ERROR;
    e=*S;
    return OK;
}
//遍历
void f(LinkMEnode S){
    MEnode *p;
    p=S;
    int i=1;
    while(p!=NULL){
        cout<<"第"<<i<<"个药品的编号为:"<<p->id<<endl;
        cout<<"第"<<i<<"个药品的名称为:"<<p->name<<endl;
        cout<<"第"<<i<<"个药品的库存为:"<<p->number<<endl;
        i++;
        p=p->next;
    }
}
//库存变化函数
void add(LinkMEnode &S,string ch,int n){
    MEnode *p;
    p=new MEnode();
    p=S;
    int f=0;
    while(p!=NULL){
        if(ch==p->id){
            p->number+=n;
            return;
        }
        p=p->next;
    }
    cout<<"输入错误"<<endl;
}
int main(){
    LinkMEnode S;
    InitStack(S);
    //输入内容
    int n;
    cin>>n;
    MEnode e;
    for(int i=1;i<=n;i++){
        cout<<"请输入第"<<i<<"个药品的编号:";
        cin>>e.id;
        cout<<"请输入第"<<i<<"个药品的名称:";
        cin>>e.name;;
        cout<<"请输入第"<<i<<"个药品的库存:";
        cin>>e.number;
        Push(S,e);//将药品信息压入栈
    }
    cout<<endl;
    int caozuo;
    cout<<"有以下操作可选择,插入请输入1,删除请输入2,查找请输入3,取值(最上端)请输入4,无操作请输入0."<<endl;
    cout<<"请输入:";
    cin>>caozuo;
    //插入
    if(caozuo==1){
        MEnode e;
        cout<<"请输入药品的编号:";
        cin>>e.id;
        cout<<"请输入药品的名称:";
        cin>>e.name;
        cout<<"请输入药品的库存:";
        cin>>e.number;
        if(Push(S,e)){
            printf("插入成功!\n");
        }
        else{
            cout<<"插入失败."<<endl;
        }
    }
    //删除
    if(caozuo==2){
        MEnode e;
        if(pop(S,e)){
            cout<<"删除成功!"<<endl;
            cout<<"删除的元素的编号为:"<<e.id<<endl;
        }
        else{
            cout<<"删除失败!"<<endl;
        }
    }
    //查找
    if(caozuo==3){
        string ch;
        cout<<"请输入要查找的编号:";
        cin>>ch;
        MEnode e;
        if(check(S,ch,e)){
            cout<<"查找成功!"<<endl;
            cout<<"药品的编号为:"<<e.id<<endl;
            cout<<"药品的名称为:"<<e.name<<endl;
            cout<<"药品的库存为:"<<e.number<<endl;
        }
        else{
            cout<<"查找失败"<<endl;
        }
    }
    //取值
    if(caozuo==4){
        MEnode e;
        if(Get(S,e)){
            cout<<"取值成功!";
            cout<<"药品的编号为:"<<e.id<<endl;
            cout<<"药品的名称为:"<<e.name<<endl;
            cout<<"药品的库存为:"<<e.number<<endl;
        }
        else{
            cout<<"取值失败!"<<endl;
        }
    }
    //遍历
    cout<<endl;
    cout<<"剩余药品如下"<<endl;
    f(S);
    //库存变化
    int change;
    cout<<"库存有变化请输入1,无变化请输入2:";
    cin>>change;
    if(change==1){
        int num;
        string ch;
        cout<<"请输入库存需变化的编号:";
        cin>>ch;
        cout<<"请输入变化数量:";
        cin>>num;
        add(S,ch,num);
    }
    //遍历
    cout<<endl;
    cout<<"剩余药品如下:"<<endl;
    f(S);
    return 0;
}