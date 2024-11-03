#include<stdio.h>
int main()
{
	int i,n,y=1,sum=0;
	scanf("%d",&n);
	for(i=1;i<=n;i++)
	{
		y*=i;
		sum+=y;
		
	}
	printf("%d",sum);
	
	return 0;
}
