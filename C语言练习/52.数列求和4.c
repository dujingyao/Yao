#include<stdio.h>
int main()
{
	int n,a,i,sum=0,b=1;
	scanf("%d %d",&n,&a);
	int x=a;
	for(i=1;i<=n;i++)
	{
		sum+=a;
		b*=10;
		a=a+x*b;
		
	}
	printf("%d",sum);
	
	return 0;
}
