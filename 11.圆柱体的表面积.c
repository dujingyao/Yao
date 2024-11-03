#include<stdio.h>

#define PI 3.14159
int main()
{
	double r,h;
	scanf("%lf %lf",&r,&h);
	double s=2*PI*r*r+2*PI*r*h;
	printf("%.2lf",s);
	
	return 0;
}
