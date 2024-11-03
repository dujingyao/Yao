#include<stdio.h>
int main()
{
	int i,x,max=0;
	for(i=1;;i++)
	{
		scanf("%d",&x);
		if(x>=max)
		{	
			max=x;
		}
		if(x<0)
		{printf("%d",max);	
		break;}
	}
	return 0;
}
