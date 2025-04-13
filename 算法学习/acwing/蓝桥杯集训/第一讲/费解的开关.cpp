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
        if(backup[a][b]=='1') backup[a][b]='0';
        else backup[a][b]='1';
    }
}

int main(){
    int T;
    cin>>T;
    while(T--){
        for(int i=0;i<5;i++){
            cin>>flag[i];
        }
        int max1=10;
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
            for(int i=1;i<5;i++){
                for(int j=0;j<5;j++){
                    if(backup[i-1][j]=='0'){
                        turn(i,j);
                        step++;
                    }
                }
            }
            bool dark=false;
            for(int i=0;i<5;i++){
                if(backup[4][i]=='0'){
                    dark=true;
                    break;
                }
            }
            if(!dark){
                max1=min(step,max1);
            }
        }
        if(max1<=6) cout<<max1<<endl;
        else cout<<-1<<endl;
    }

    return 0;
}