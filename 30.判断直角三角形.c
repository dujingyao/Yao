#include<stdio.h>
int main()
{
	int a,b,c;
	scanf("%d %d %d",&a,&b,&c);
	if(a+b>c&&a+c>b&&c+b>a&&(a*a+b*b==c*c||a*a+c*c==b*b||b*b+c*c==a*a))
		printf("Yes");
	else printf("No");
	
	return 0;
}
