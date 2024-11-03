#include<stdio.h>
#include<math.h>
int main()
{
	int x,y;
	scanf("%d",&x);
	int z=abs(3*x+2);
	if(x<-2)
		y=7-2*x;
	if(x>=-2&&x<3)
		y=5-z;
	if(x>=3)
		y=3*x+4;
	printf("%d",y);
	
	return 0;
}
