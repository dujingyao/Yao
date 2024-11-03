#include<stdio.h>
int main()
{
	int n,i;
	double y=0.0,f=1.0,a=1.0,b=1.0;
	scanf("%d",&n);
	for(i=1;i<=n;i++)
	{
		y+=f*a/b;
		f=-f;
		a++;
		b+=2;
	}
	printf("%.3lf",y);
	
	return 0;
}
