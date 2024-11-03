#include<stdio.h>
int main()
{
	int a;
	double b;
	scanf("%d",&a);
	if(a<500)
		b=a;
	if(a>=500&&a<1000)
		b=0.95*a;
	if(a>=1000&&a<3000)
		b=0.9*a;
	if(a>=3000&&a<5000)
		b=0.85*a;
	if(a>=5000)
		b=0.8*a;
	printf("%.2lf",b);
	return 0;
}
