#include<stdio.h>
int main()
{
	int n,y,i,a=1;
	scanf("%d",&n);
	for(i=1;i<=n-1;i++)
	{
		y=(a+1)*2;
		a=y;
	}
	printf("%d",y);
	
	return 0;
}
