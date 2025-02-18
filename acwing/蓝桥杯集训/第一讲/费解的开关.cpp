#include<iostream>
#include<algorithm>
#include <cstdio>
#include <cstring>
using namespace std;

char flag[30][30],backup[30][30];

int dx[5]={-1,0,1,0,0},dy[5]={0,1,0,-1,0};

void turn(int x,int y){
    for(int i=0;i<5;i++){
        int a=x+dx[i],b=y+dy[i];
    }
}

int main(){
    int T;
    cin>>T;
    while(T--){
        for(int i=0;i<5;i++){
            cin>>flag[i];
        }
        for(int op=0;op<32;op++){
            memcpy(backup,flag,sizeof flag);
            int step=0;
            //遍历第一行所有
            for(int i=0;i<5;i++){
                if(op>>i&1){
                    step++;
                    turn(0,i);
                }
            }
        }

    }

    return 0;
}