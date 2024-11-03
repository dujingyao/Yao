#include<stdio.h>
int main()
{
	int n,i;
	double m=1.0,sum=0.0;
	scanf("%d",&n);
	for(i=1;;i++)
	{
		m*=i;
		sum+=m;
		if(sum>=n)
		{
			printf("m<=%d",i-1);
			break;
		}
	}
	return 0;
}
