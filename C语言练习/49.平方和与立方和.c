#include<stdio.h>
int main()
{
	int m,n,i,x=0,y=0;
	scanf("%d %d",&m,&n);
	for(i=m;i<=n;i++)
	{
		if(i%2==1)
			x+=i*i*i;
		if(i%2==0)
			y+=i*i;
		
	}
	printf("%d %d",y,x);
	return 0;
}
