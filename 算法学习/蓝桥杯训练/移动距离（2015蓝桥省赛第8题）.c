#include<stdio.h>
#include<math.h>
int func_col(int row,int n,int sum){
    if(row%2==1) return sum-(row-1)*n;
    if(row%2==0) return 2*n-sum+1;
}
int main(){
    int n;
    int x,y;
    scanf("%d %d %d",&n,&x,&y);
    int x_row,x_col,y_row,y_col;
    x_row=x/(n+1)+1,y_row=y/(n+1)+1;
    x_col=func_col(x_row,n,x);
    y_col=func_col(y_row,n,y);
    printf("%d",abs((x_row-y_row)+(x_col-y_col)));
    return 0;
}