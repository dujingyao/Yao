#include<stdio.h>
int main()
{
	int n,i,f=0,l=0,z=0;
	double a;
	scanf("%d",&n);
	for(i=1;i<=n;i++)
	{
		scanf("%lf",&a);
		if(a<0)
			f++;
		if(a>0)
			z++;
		if(a==0)
			l++;
	}printf("%d %d %d",f,l,z);
	
	return 0;
}
