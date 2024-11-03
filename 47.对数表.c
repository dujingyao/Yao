#include<stdio.h>
#include<math.h>
int main()
{
	double a;
	int m,n,i;
	scanf("%d %d",&m,&n);
	for(i=m;i<n+1;i++)
	{
		a=log(i);
		printf("%4d%8.4lf\n",i,a);
	}
	return 0;
}
