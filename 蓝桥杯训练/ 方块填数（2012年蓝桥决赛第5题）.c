#include<stdio.h>
char a[6][6]={{0}};
void panduan(int row,int col){
    //判断行
    for(int i=0;i<6;i++){
        if(i==col) continue;
        if(a[row][col]==a[row][i]) return;
    }
    //判断列
    for(int i=0;i<6;i++){
        if(i==row) continue;
        if(a[row][col]==a[i][col]) return;
    }
    //判断同色块
    if(row>=0&&row<=3&&&col==0||row==1&&(col==0||col==3)){
        for(int i=0;i<=3;i++){
            if(row==0&&col==i) continue;
            if(a[row][col]==a[0][i]) return;
        }
        if(row==1&&col==0) ;
        else if(a[row][col]==a[1][0]) return;
        if(row==1&&col==3) ;
        else if(a[row][col]==a[1][3]) return;
    }
    else if(row>=1&&row<=2&&col==1||row>=0&&row<=1&&col==2||row==0&&col>=3&&col<=4){
        
    }
}
void huisu(int a){

}

int main(){
    int x,y,n;
    scanf("%d",&n);
    for(int i=1;i<=n;i++){
        scanf("%d %d %c",&x,&y,&a[x][y]);
    }

    return 0;
}