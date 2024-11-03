#include<stdio.h>
int main()
{
	double a;
	scanf("%lf",&a);
	double b=-a;
	if(a>=0)
		printf("%.2lf",a);
	else
		printf("%.2lf",b);
	
	return 0;
}
