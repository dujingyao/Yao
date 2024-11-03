#include<stdio.h>
int main()
{
	int m,i,j,n;
	double a[1001][11],x;
	double p;
	scanf("%d %d",&m,&n);
	for(i=0;i<m;i++){
		for(j=0;j<n;j++){
			scanf("%lf",&a[i][j]);
		}
	}
	for(j=0;j<n;j++){
		for(i=0;i<m;i++){
			x=x+a[i][j];
		}
		p=1.0*x/m;
		printf("%.2f ",p);
		p=0,x=0;
	}
	
	return 0;
}
