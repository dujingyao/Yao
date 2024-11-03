#include<stdio.h>
int main()
{
	int a,b,c,d,e,f;
	scanf("%d:%d:%d\n %d:%d:%d",&a,&b,&c,&d,&e,&f);
	int x=d*3600+e*60+f-a*3600-b*60-c;
	printf("%d",x);
	
	return 0;
}
