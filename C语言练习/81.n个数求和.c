#include<stdio.h>
int main()
{
	int T,i;
	scanf("%d",&T);
	for(i=1;i<=T;i++)
	{
		int n,j,c=0,x;
		scanf("%d",&n);
		for(j=1;j<=n;j++)
		{
			scanf("%d",&x);
			c=c+x;
		}
		printf("%d\n",c);
	}
	
	return 0;
}
