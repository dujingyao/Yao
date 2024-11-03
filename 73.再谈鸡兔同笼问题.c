#include<stdio.h>
#include<math.h>
int main()
{
	int m,n;
	double x,y;
	scanf("%d %d",&m,&n);
	x=(4*m-n)*1.0/2;
	y=(n-2*m)*1.0/2;
	if(floor(x)<x||floor(y)<y||x<0||y<0)
		printf("No Answer");
	else printf("%.0f %.0f",x,y);
	
	return 0;
}
