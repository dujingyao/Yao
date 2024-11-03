#include<stdio.h>
int main()
{
	int x,i,y,m,d;
	scanf("%d",&x);
	for(i=1;i<=x;i++)
	{
		scanf("%*6d%4d%2d%2d%*d",&y,&m,&d);
		printf("%d-%02d-%02d\n",y,m,d);
	}
	
	return 0;
}
