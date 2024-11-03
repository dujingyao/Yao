#include<stdio.h>
#include<math.h>
int main()
{
	double a,b,c;
	scanf("%lf %lf %lf",&a,&b,&c);
	double p=(a+b+c)/2,s=sqrt(p*(p-a)*(p-b)*(p-c));
	printf("%.2lf",s);
	
	return 0;
}
