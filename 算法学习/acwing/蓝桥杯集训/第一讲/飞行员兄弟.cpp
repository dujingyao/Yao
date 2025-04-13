#include<iostream>
#include<vector>
#include<algorithm>
#include<cstring>
#include<cstdio>
using namespace std;

typedef pair<int,int> PII;

const int N=5;

char g[N][N],backup[N][N];

int get(int x,int y){
    return x*4+y;
}

void turn_one(int x,int y){
    if(backup[x][y]=='+') backup[x][y]='-';
    else backup[x][y]='+';
}

void turn_all(int x,int y){
    for(int i=0;i<4;i++){
        turn_one(x,i);
        turn_one(i,y);
    }
    turn_one(x,y);
}

int main(){
    for(int i=0;i<4;i++)
        for(int j=0;j<4;j++)    
            cin>>g[i][j];
    vector<PII> res;//记录方案所需要的结构
    for(int op=0;op<1<<16;op++){
        vector<PII> temp;//存方案
        memcpy(backup,g,sizeof g);
        for(int i=0;i<4;i++){
            for(int j=0;j<4;j++){
                if(op>>get(i,j)&1){
                    temp.push_back({i,j});
                    turn_all(i,j);
                }
            }
        }
        bool has_closed=false;
        for(int i=0;i<4;i++){
            for(int j=0;j<4;j++){
                if(backup[i][j]=='+') has_closed=true;
            }
        }
        if(has_closed==false){
            //如果是第一轮，那么方案就为空
            if(res.empty()||res.size()>temp.size()) res=temp;
            cout<<res.size()<<endl;
            for(auto op:res) cout<<op.first+1<<' '<<op.second+1<<endl;
        }
    }
    return 0;
}