#include<stdio.h>
int main()
{
	int n,a,i,m=1;
	scanf("%d",&n);
	for(i=1;i<=n;i++)
	{
		scanf("%d",&a);
		if(a%2==1)
			m*=a;	
	}printf("%d",m);
	
	return 0;
}
