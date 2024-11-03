#include<stdio.h>
int main()
{
	int n,x;
	double y=0;
	scanf("%d",&n);
	for(x=1;x<=2*n-1;x+=2)
	{
		y=y+1/(x*1.0);
	}
	printf("%.2lf",y);
		
	
	return 0;
}
