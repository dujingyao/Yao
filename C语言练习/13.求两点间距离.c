#include<stdio.h>
#include<math.h>
int main()
{
	double x1,y1,x2,y2;
	0>=x1,y1,x2,y2>=100;
	scanf("%lf %lf %lf %lf",&x1,&y1,&x2,&y2);
	double d=sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
	printf("%.2lf",d);
	
	
	return 0;
}
