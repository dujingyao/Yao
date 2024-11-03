#include<stdio.h>
int a[1000000];
int main()
{
	int n,m,max,sum=0;
	scanf("%d",&n);
	int i,j;
	for(i=1;i<=n;i++){
		scanf("%d",&m);
		for(j=0;j<m;j++){
			scanf("%d",&a[j]);
		}
		max=a[0];
		for(j=0;j<m;j++){
			if(a[j]>max) max=a[j];
		}
		for(j=0;j<m;j++){
			if(a[j]!=max) sum=sum+a[j];
		}
		if(sum>=max-1) printf("Yes");
		else printf("No");
		printf("\n");
		sum=0;
	}
	
	return 0;
}
