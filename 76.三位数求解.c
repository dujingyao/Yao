#include<stdio.h>
int main()
{
	int n,x,y,z,flag=0;
	scanf("%d",&n);
	for(x=0;x<=9;x++)
	{
		for(y=0;y<=9;y++)
		{
			for(z=0;z<=9;z++)
			{
				if(x!=0&&y!=0&&x*100+y*10+z+y*100+z*10+z==n)
				{
					flag=1;
					printf("%4d%4d%4d",x,y,z);
				}
			}
		}
	}
	if(flag==0)
	{
		printf("No Answer");
	}
	
	return 0;
}
