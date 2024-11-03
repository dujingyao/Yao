#include<stdio.h>
int main()
{
	int i,n,m,t;
	m=0;
	scanf("%d\n",&n);
	for(i=1;i<=n;i++)
	{
		scanf("%d",&t);
		m=m+t;
	}
	printf("%d",m);
	
	return 0;
}
