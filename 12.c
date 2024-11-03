#include<stdio.h>
#include<string.h>
int KToD(char str[],int k)
{
	int i,m,x=1,sum=0;
	m=strlen(str);
	for(i=m-1;i>=0;i--)
	{
		sum+=(str[i]-'0')*x;
		x=x*k;//k进制 
	}
	return sum;
}
int main()
{
	int n,m,i,k,a[100000],max;
	char str[100000];
	scanf("%d",&n);
	for(i=0;i<n;i++)
	{
		scanf("%s",str);
		getchar();
		scanf("%d",&k);
		a[i]=KToD(str,k);//将每一个值存入数组中 
	}
	max=a[0];
	for(i=1;i<n;i++)
	{
		if(a[i]>max)  
			max=a[i];
	} 
	printf("%d",max);
	return 0;
}
