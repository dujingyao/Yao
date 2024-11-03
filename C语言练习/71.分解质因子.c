#include<stdio.h>
int main()
{
	int n;
	scanf("%d",&n);
	for(int i=2;i<=n;i++)
	{
		while(n%i==0)
		{
			n=n/i;
			printf("%d ",i);
		}
	}
	
	return 0;
}
