#include<stdio.h>
int main()
{
	int n,i;
	scanf("%d",&n);
	int A,B;
	for(i=1;i<=n;i++)
	{
		int y=1;
		scanf("%d %d",&A,&B);
		for(int j=1;j<=B;j++)
		{
			y=y*A;
			y=y%1000;
		}
		printf("%d\n",y);
	}
	
	return 0;
}
