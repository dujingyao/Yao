#include<stdio.h>
int main()
{
	int M,k,day=0;
	scanf("%d %d",&M,&k);
	while(M>0)
	{
		M--;
		day++;
		if(day%k==1&&day!=1)
		{
			M++;
		}
	}
	if(M==0&&day%k==0)
		day++;
		
	printf("%d",day);
	return 0;
}
