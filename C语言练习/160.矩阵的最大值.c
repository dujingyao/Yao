#include<stdio.h>
void FindMax(int p[][3], int m, int n, int *pRow, int *pCol);
int main(){
    int a[2][3],i,j;
    for(i=0;i<2;i++){
        for(j=0;j<3;j++){
            scanf("%d",&a[i][j]);
        }
    }
	int *pRow=&i,*pCol=&j;
    FindMax(a,2,3,pRow,pCol);
    return 0;
}
void FindMax(int p[][3], int m, int n, int *pRow, int *pCol){
    int i,j,k=0;
    int max=p[0][0];
    for(i=0;i<m;i++){
        for(j=0;j<n;j++){
            if(max<p[i][j]) max=p[i][j];
        }
    }
    for(i=0;i<m;i++){
        for(j=0;j<n;j++){
            if(max==p[i][j]){
                pRow=&i;
                pCol=&j;
                k=1;
                break;
            }
        }
		if(k==1) break;
    }
    printf("%d %d %d",max,*pRow,*pCol);

}