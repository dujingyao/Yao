#include<bits/stdc++.h>
using namespace std;

const int N=1e5+10;
int w[N];

int n,m;

struct node{
    int l,r;
    int maxv;
}tr[N*4];

//构造线段树
void build(int u,int l,int r){//u代表当前节点
    if(l==r) tr[u]={l,r,w[r]};
    else{
        tr[u]={l,r};
        int mid=l+r>>1;
        build(u<<1,l,mid),build(u<<1|1,mid+1,r);
        tr[u].maxv=max(tr[u<<1].maxv,tr[u<<1|1].maxv);
    }
}
// 查询操作
int query(int u,int l,int r){
    if(tr[u].l>=l&&tr[u].r<=r) return tr[u].maxv;
    int mid=tr[u].l+tr[u].r>>1;
    int max1=INT_MIN;
    if(l<=mid) max1=query(u<<1,l,r);//查询左子树的最大值
    if(r>mid) max1=max(max1,query(u<<1|1,l,r));//查询右子树的最大值
    return max1;
}

int main(){
    scanf("%d %d",&n,&m);
    for(int i=1;i<=n;i++) scanf("%d",&w[i]);
    build(1,1,n);
    while(m--){
        int x,y;
        scanf("%d %d",&x,&y);
        printf("%d\n",query(1,x,y));
    }

    return 0;
}