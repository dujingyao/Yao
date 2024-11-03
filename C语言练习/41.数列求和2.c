#include<stdio.h>
int main()
{
	int i,n;
	double s=0.0,a=1.0,f=1.0;
	scanf("%d",&n);
	for(i=1;i<=n;i++)
	{
		s+=f/a;
		a+=2;
		f=-f;
	}		
	printf("%.2lf",s);
	
	
	
	return 0;
}
