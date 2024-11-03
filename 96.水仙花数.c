#include<stdio.h>
int narcissus(int n)
{
	int b,s,g;
	b=n/100;
	s=n%100/10;
	g=n%100%10;
	if(n==b*b*b+s*s*s+g*g*g)
	{
		return 1;
	}else return 0;
}
int main()
{
	int m,n;
	while(scanf("%d %d",&m,&n)!=EOF)
	{
		int i,j=0;
		for(i=m;i<=n;i++)
		{
			if(narcissus(i)) 
			{
				j++;
				printf("%d ",i);
			}
		}
		if(j==0)
		{
			printf("no");
		}printf("\n");
	}
	
	return 0;
}
