#include<stdio.h>
#define N 100
int f(int a[][N],int n);
int main()
{
	int a[100][100];
	int n,i,j,x=0,y=0,m=0,k=0;
	scanf("%d",&n);
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			scanf("%d",&a[i][j]);
		}
	}
	if(f(a,n)==1) {
		k=1;
		printf("OK");
	}
	else{
		for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			if(a[i][j]==0){
				a[i][j]=1;
				if(f(a,n)==1){
					m++;
					x=i,y=j;
				}
				a[i][j]=0;
			}
			if(a[i][j]==1){
				a[i][j]=0;
				if(f(a,n)==1){
					m++;
					x=i,y=j;
				}
				a[i][j]=1;
			}
			}
		}
	}
	if(m==1){
		k=1;
		printf("Change bit(%d,%d)",x,y);
	}
	if(k==0){
		printf("Corrupt");
	}
	
	return 0;
}
int f(int a[][N],int n){
	int i,j,x=0,y=0,m=0,h=0;
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			if(a[i][j]==1) x++;
		}
		if(x==0||x%2==0) y++;
		x=0;
	}
	for(j=0;j<n;j++){
		for(i=0;i<n;i++){
			if(a[i][j]==1) m++;
		}
		if(m==0||m%2==0) h++;
		m=0;
	}
	if(y==n&&h==n) return 1;
	else return 0;
}
