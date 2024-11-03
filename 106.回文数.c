#include<stdio.h>
int f(int n)
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
	int m,n;
	scanf("%d %d",&m,&n);
	int i;
	for(i=m;i<=n;i++)
	{
		if(i==f(i)) printf("%d ",i);
	}
	
	return 0;
}
