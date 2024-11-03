#include<stdio.h>
int main()
{
	long m,n;
	scanf("%ld %ld",&m,&n);
	long a=m,b=n;
	while(1)
	{
		if(m>n) m=m-n;
		else if(n>m) n=n-m;
		else break;
	} 
	printf("%ld %ld",n,(a*b)/n);
	return 0;
}
