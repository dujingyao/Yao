#include<stdio.h>
int digitSum(int n)
{
	int x=0,a=n;
	while(n>0)
	{
		n=n/10;
		x++;
	}
	int m,h;
	while(1)
	{
		m=0;
		for(;x>0;x--)
		{
			h=a%10;
			m+=h;
			a=a/10;
		}
		int j=m;
		while(j>0)
		{
			j=j/10;
			x++;
		}
		a=m;
		if(m<10) break;
	}
	return m;
}
int main()
{
	int n;
	scanf("%d",&n);
	printf("%d",digitSum(n));
	
	return 0;
}
