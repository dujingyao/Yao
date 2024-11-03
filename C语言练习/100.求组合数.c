#include<stdio.h>
int fact(int n)
{
	int i;
	int sum=1;
	for(i=1;i<=n;i++)
	{
		sum=i*sum;
	}
	return sum;
}
int main()
{
	int m,k;
	scanf("%d %d",&m,&k);
	int x;
	x=fact(m)/(fact(k)*fact(m-k));
	printf("%d",x);
	
	return 0;
}
