#include<bits/stdc++.h>
using namespace std;
const int N=210;
int k[N];//每层电梯的对应上下数字
int s[2]={-1,1};
queue<int> q;
bool st[N];//标记当前楼层是否来过
int n,a,b,g;
int ans[N];//按了几次
void bfs(int f){
    q.push(f);
    st[f]=true;
    while(!q.empty()){
        auto h=q.front();
        q.pop();
        for(int i=0;i<2;i++){
            int w=h+s[i]*k[h];//操作后的楼层
            if(w>n||w<1) continue;
            if(st[w]) continue;
            st[w]=true;
            ans[w]=ans[h]+1;
            q.push(w);
            if(b==w){
                g=1;
                return;
            }
        }
    }
}
int main(){
    cin>>n>>a>>b;
    for(int i=1;i<=n;i++) cin>>k[i];
    if(a==b){
        cout<<0<<endl;
        return 0;
    }
    bfs(a);
    if(g==0){
        cout<<-1<<endl;
    }
    else cout<<ans[b]<<endl;
    return 0;
}