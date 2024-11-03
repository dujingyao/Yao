#include<stdio.h>
int main()
{
	int n,i,j,m=2;
	int a[31][31];
    scanf("%d",&n);
	for(j=0;j<n+2;j++){
		a[0][j]=0;
	}
	for(i=0;i<n;i++){
		a[i][0]=0;
	}
	a[0][1]=1;
	for(i=1;i<n;i++){
		for(j=1;j<n+2;j++){
			a[i][j]=a[i-1][j-1]+a[i-1][j];
		}
	}
	for(i=0;i<n;i++){
		for(j=1;j<m;j++){
			printf("%d ",a[i][j]);
		}
		printf("\n");
		m++;
		if(m>n+1) break;
	}

	return 0;
}

