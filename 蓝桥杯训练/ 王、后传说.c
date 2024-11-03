#include<stdio.h>
#include <string.h>
#include<math.h>
int count=0;
int n,x,y;
int a[12][12];
int panduan(int row,int col){//行,列
    //检查列
    for(int i=0;i<row;i++){
        if(a[i][col]==1) return 0;
    }
    //检查主对角线
   for(int i=row,j=col;i>=0&&j>=0;i--,j--) {
        if(a[i][j]==1) return 0;
    }
    // 检查副对角线
    for(int i=row,j=col;i>=0&&j<n;i--,j++) {
        if (a[i][j] == 1) return 0;
    }
    //检查国王
    if ((row>=x-1&&row<=x+1)&&(col>=y-1&&col<=y+1)) {
        return 0;
    }
    return 1;
}
void huisu(int row){
    if(row==n){
        count++;
        return;
    }
    for(int i=0;i<n;i++){
        if(panduan(row,i)){
            a[row][i]=1;
            huisu(row+1);
            a[row][i]=0;
        }
    }
}
int main(){
    scanf("%d %d %d",&n,&x,&y);
    x=x-1,y=y-1;
    memset(a,0,sizeof(a));
    huisu(0);
    printf("%d",count);

    return 0;
}