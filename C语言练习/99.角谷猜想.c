#include<stdio.h>
int f(int x)
{
	int i;
	for(i=0;x!=1;i++)
	{
		if(x%2==0) x=x/2;
		else x=x*3+1;
	}
	return i;
}
int main()
{
	int n;
	int y;
	while(scanf("%d",&n)!=EOF)
	{
		y=f(n);
		printf("%d\n",y);
	}
	
	return 0;
}
