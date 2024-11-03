#include<stdio.h>
int inverse(int n)
{
	int i=0,a=n,x;
	while(n>0)
	{
		i++;
		n=n/10;
	}
	int j=i,h=i,sum=0;
	for(;i>0;i--)
	{
		x=a%10;
		a=a/10;
		for(;j>1;j--)
		{
			x=x*10;
		}
		sum+=x;
		h--;
		j=h;
	}
	return sum;
}
int main()
{
	int n,m;
	scanf("%d",&n);
	while(m=inverse(n),m!=n)
	{
		printf("%d ",n);
		n=m+n;
	}
	printf("%d",m);
	
	return 0;
}
