#include<stdio.h>
int inverse(int t)
{
	int i;
	int a=t;
	int x,m=0;
	for(i=0;t>0;i++)
	{
		t=t/10;
	}
		while(a>0)
	{
		x=a%10;
		a=a/10;
		int j=i;
		for(j=i;j>1;j--)
		{
			x=x*10;
		}
		i--;
		m=x+m;
	}
	return m;
}
int main()
{
	int n,  sum;
	scanf("%d", &n);
	sum = n + inverse(n);
	printf("%d", sum);  
	
	return 0;
}
