#include<stdio.h>
#include<math.h>
int main()
{
	int n;
	double item,i,sum=0.0;
	scanf("%lf %d",&item,&n);
	for(i=1;i<=n;i++)
	{
		sum+=item;
		item=sqrt(item);
	}
	printf("%.2lf",sum);
	
	return 0;
}
