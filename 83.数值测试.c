#include<stdio.h>
int main()
{
	int n,i,a=0,b=0,c=0;
	double j;
	while(scanf("%d",&n),n!=0)
	{
		for(i=1;i<=n;i++)
		{
		scanf("%lf",&j);
		if(j<0) a++;
		else if(j==0) b++;
		else if(j>0) c++;
		}
		printf("%d %d %d\n",a,b,c);
		a=b=c=0;
		
	}
	return 0;
}
