#include<stdio.h>
int main()
{
	int a,c,i,b;
	scanf("%d",&a);
	scanf("%d",&b);
	for(i=1;i<a;i++)
	{
		scanf("%d",&c);
		if(c>=b)
			b=c;
	}printf("%d",b);
	
	
	return 0;
}
