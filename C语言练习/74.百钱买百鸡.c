#include<stdio.h>
int main()
{
	int i,j,flag=0;
	int n;
	scanf("%d",&n);
	for(i=0;i<=n/5;i++)
	{
		for(j=0;j<=n/3;j++)
		{
			if(i*5+j*3+(n-i-j)/3.0==n)
			{
				flag=1;
				printf("%4d%4d%4d\n",i,j,n-i-j);
			}
		}
	}
	if(flag==0)
	{
		printf("No Answer");
	}
	
	return 0;
}
