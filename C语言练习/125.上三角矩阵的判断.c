#include<stdio.h>
#define N 15
int IsUpperTriMatrix(int a[][N],int n);
int main()
{
	int a[15][15];
	int b,i,j,m;
	scanf("%d",&b);
	for(i=0;i<b;i++)
	for(j=0;j<b;j++)
		scanf("%d",&a[i][j]);
	m=IsUpperTriMatrix(a,b);
	if(m==0) printf("YES");
	if(m==1) printf("NO");
	
	return 0;
}
int IsUpperTriMatrix(int a[][N],int n){
	int i,j;
	for(i=1;i<n;i++)
		for(j=1;j<i;j++){
			if(a[i][j]!=0)
				return 0;
		}
	return 1;
}
