#include<bits/stdc++.h>
using namespace std;

const int N=1010;
int n,m;
int d[N][N];
int fx,fy,lx,ly,per;//per记录周长

void update(int x,int y){
    int dx=x-lx,dy=y-ly;
    int step=abs(dx)+abs(dy);
    dx/=step,dy/=step;//将方向归一化
    //dx,dy表示每一步的方向增量
    while(step--){
        lx+=dx,ly+=dy;
        d[lx][ly]=++per;//每多走一步，周长加一
    }
}

int main(){
    
    cin>>n>>m>>fx>>fy;
    lx=fx,ly=fy;
    while(--m){
        int x,y;
        cin>>x>>y;
        update(x,y);
    }
    update(fx,fy);//把最后一个点和第一个点连起来
    while(n--){
        int x1,y1,x2,y2;
        cin>>x1>>y1>>x2>>y2;
        int dis=abs(d[x1][y1]-d[x2][y2]);
        cout<<min(dis,per-dis)<<endl;
    }

    return 0;
}