#include<stdio.h>
int main()
{
	int n,i;
	long y=1;
	scanf("%d",&n);
	for(i=1;i<=n;i++)
	{
		y*=i;
		printf("%-4d%-20ld\n",i,y);
	}
	
	return 0;
}
