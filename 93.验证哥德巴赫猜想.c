#include<stdio.h>
#include<math.h>
int prime(int n)
{
	int i,k;
	if(n==1) return 0;
	k=(int)sqrt(n);
	for(i=2;i<=k;i++)
	{
		if(n%i==0) return 0;
	}
	return 1;
}
int main()
{
	int M,j;
	scanf("%d",&M);
	for(j=1;j<=M/2;j++)
	{
		if(prime(j))
		{
			if(prime(M-j))
			{
				printf("%d %d\n",j,M-j);
			}
		}
	}
	
	return 0;
}
