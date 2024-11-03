#include<stdio.h>
int main()
{
	int num,i,x=0;
	scanf("%d",&num);
	for(i=num;i>=1;i--)
	{
		if(i%10!=4&&i/10%10!=4&&i/100%10!=4)
		{
			x++;
		}
	}
	printf("%d",x);
	
	return 0;
}
