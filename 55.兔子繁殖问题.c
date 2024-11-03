#include<stdio.h>
int main()
{
	int i,n;
	int preNum,curNum,temp;
	scanf("%d",&n);
	preNum=curNum=1;
	for(i=3;i<=n;i++)
	{
		temp=curNum;
		curNum=preNum+curNum;
		preNum=temp;
	}
	printf("%d",curNum);
	
	return 0;
}
