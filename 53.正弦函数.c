#include<stdio.h>
int main()
{
	int t,i,n;
	double y=0.0,xiang=1,fuhao,x;
	scanf("%lf",&x);
	for(t=1;t<=10;t++)
	{
		double fenzi=1.0,fenmu=1.0;
		if(t%2==1)
			fuhao=1;
		else
			fuhao=-1;
		for(i=1;i<=xiang;i++)
		{
			fenzi*=x;
		}
		for(n=1;n<=xiang;n++)
		{
			fenmu*=n;
		}
		xiang=xiang+2;
		y=y+fuhao*fenzi/fenmu;
	}
	printf("%.3lf",y);

	return 0;
}
