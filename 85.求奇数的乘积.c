#include<stdio.h>
int main()
{
	int n,i;
	int x,y=1;
	while(scanf("%d",&n)!=EOF)
	{
		y=1;
		for(i=0;i<n;i++)
		{
		scanf("%d",&x);
		if(x%2!=0)
		{
			y=y*x;
		}
		}
		printf("%d\n",y);
	}
	
	return 0;
}
