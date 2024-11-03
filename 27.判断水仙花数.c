#include<stdio.h>
int main()
{
	int a;
	scanf("%d",&a);
	int b=a/100,c=a%100/10,d=a%100%10;
	if(a==b*b*b+c*c*c+d*d*d)
		printf("yes");
	else printf("no");
	return 0;
}
