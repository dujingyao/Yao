#include<stdio.h>
#include<math.h>
int prime(int x)
{
	int i,k;
	if(x==1) return 0;
	k=(int)sqrt(x);
	for(i=2;i<=k;i++)
	{
		if(x%i==0) return 0;
	}
	return 1;
}
int main()
{
	int j,x,y;
	scanf("%d %d",&x,&y);
	for(j=x;j<=y;j++)
	{
		if(prime(j)) printf("%d ",j);
	}
	
	return 0;
}
