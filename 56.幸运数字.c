#include<stdio.h>
int main()
{
	int m,n,i,x=0;
	scanf("%d %d",&m,&n);
	for(i=n;i>=m;i--)
	{
		if(i%7==0&&i%4!=0)
		{	x=1;
			break;
		}	
	}if(x==0)
		printf("no");
	else printf("%d",i);
	
	return 0;
}
