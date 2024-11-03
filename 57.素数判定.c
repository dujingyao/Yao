#include<stdio.h>
int main()
{
	int n,i,x=0;
	scanf("%d",&n);
	if(n<=1)
		x=1;
	else
	for(i=n-1;i>=2;i--)
	{
		if(n%i==0)
		{	x=1;
			break;
		}	
	}if(x==1)
		printf("No");
	else printf("Yes");
	
	return 0;
}
