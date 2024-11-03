#include<stdio.h>
#include<math.h>
int main()
{
	double s=0.0,x1,y1,x2,y2;
	while(scanf("%lf %lf %lf %lf",&x1,&y1,&x2,&y2)!=EOF)
	{
		s=sqrt(1.0*(x1-x2)*(x1-x2)+1.0*(y1-y2)*(y1-y2));
		printf("%.2lf\n",s);
	}	
	
	return 0;
}
