#include<stdio.h>
int main()
{
	int n,cost,i,j,flag=1,x;
	scanf("%d %d",&n,&cost);
	for(i=0;i<=n;i++)
	{
		for(j=0;j<=n-i;j++)
		{
			x=n-i-j;
			if(i*3+j*2+x==cost)
			{
				flag=0;
				printf("%d %d %d\n",i,j,x);
			}
		}
	}
	if(flag==1)
		printf("No answer\n");
	
	return 0;
}
