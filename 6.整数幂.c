#include<stdio.h>
int main()
{int a,b,c;
	scanf("%d %d %d",&a,&b,&c);
	int a2=a*a,a3=a*a*a,b2=b*b,b3=b*b*b,c2=c*c,c3=c*c*c;
	printf("%-9d%-9d%-9d\n%-9d%-9d%-9d\n%-9d%-9d%-9d",a,a2,a3,b,b2,b3,c,c2,c3);
	
	return 0;
}
