#include<stdio.h>
int main()
{
	int a;
	scanf("%d",&a);
	double b;
	if(a<=10000)
		b=1500+0.05*a;
	if(a>10000&&a<=50000)
		b=1500+10000*0.05+(a-10000)*0.03;
	if(a>=50000)
		b=1500+10000*0.05+40000*0.03+(a-50000)*0.02;
		
	printf("%.2lf",b);
	
	return 0;
}
