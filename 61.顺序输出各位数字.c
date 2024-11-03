#include<stdio.h>
#include<math.h>
int main()
{
	int n,x,h=1,i;
	scanf("%d",&n);
	x=log10(n);
	for(i=1;i<=x;i++)
	{
		h*=10;
	}
	while(h>0)
	{
		printf("%d ",n/h);
		n=n%h;
		h/=10;
	}
	
	return 0;
}
