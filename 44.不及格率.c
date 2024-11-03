#include<stdio.h>
int main()
{
	int n,i,b=60;
	double a,c=0.0;
	scanf("%d",&n);
	for(i=1;i<n+1;i++)
	{
		scanf("%lf",&a);
		if(a<b)
			c++;
	}
	double r=c/n;
	printf("%.2lf",r);
	
	
}	
