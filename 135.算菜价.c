#include<stdio.h>
#include<stdlib.h>
int main()
{
	double w,p,n=0;
	while(scanf("%*s%lf%lf",&w,&p)!=EOF){
		n=n+w*p;
	}
	printf("%.1lf\n",n);
	
	return 0;
}
