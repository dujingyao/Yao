#include<stdio.h>
int main()
{
	int N,i,x;
	scanf("%d",&N);
	for(i=1;i<=N;i++)
	{
		scanf("%*6d%5d",&x);
		printf("6%05d\n",x);
	}
	
	return 0;
}
